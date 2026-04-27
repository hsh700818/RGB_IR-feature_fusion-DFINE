import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sigmoid_focal_loss
from .box_ops import rbox_to_corners, rotated_iou, rbox_to_adr_target
from ...core import register

__all__ = ["DFINECriterion"]


@register()
class DFINECriterion(nn.Module):
    __inject__ = ["matcher"]
    __share__ = ["num_classes"]

    def __init__(
        self,
        matcher,
        num_classes=5,
        weight_dict=None,
        losses=None,
        num_bins=16,
        **kwargs,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict if weight_dict is not None else {}
        self.losses = losses if losses is not None else ["labels", "rboxes"]
        self.num_bins = num_bins

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_onehot = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        target_onehot = target_onehot.to(dtype=src_logits.dtype)
        loss_ce = sigmoid_focal_loss(src_logits, target_onehot, num_boxes=num_boxes)
        return {"loss_vfl": loss_ce * self.weight_dict.get("loss_vfl", 1.0)}

    def loss_rboxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            zero = src_boxes.sum() * 0
            return {"loss_bbox": zero, "loss_riou": zero}

        src_corners = rbox_to_corners(src_boxes)
        target_corners = rbox_to_corners(target_boxes)
        loss_bbox = F.l1_loss(src_corners, target_corners, reduction="none").sum() / num_boxes

        if self.weight_dict.get("loss_riou", 0.0) > 0:
            iou_vector = rotated_iou(src_boxes, target_boxes, is_aligned=True)
            loss_riou = (1 - iou_vector).sum() / num_boxes
        else:
            loss_riou = src_boxes.sum() * 0

        return {
            "loss_bbox": loss_bbox * self.weight_dict.get("loss_bbox", 5.0),
            "loss_riou": loss_riou * self.weight_dict.get("loss_riou", 0.0),
        }

    def loss_fgl(self, outputs, targets, indices, num_boxes):
        if "pred_reg_logits" not in outputs or self.weight_dict.get("loss_fgl", 0.0) <= 0:
            pred = outputs["pred_boxes"]
            return {"loss_fgl": pred.sum() * 0}

        idx = self._get_src_permutation_idx(indices)
        src_reg_logits = outputs["pred_reg_logits"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_adr = rbox_to_adr_target(target_boxes, self.num_bins)
        loss_fgl = F.cross_entropy(
            src_reg_logits.flatten(0, 1), target_adr.flatten().long(), reduction="none"
        ).view(src_reg_logits.shape[:2])
        return {"loss_fgl": (loss_fgl.sum() / num_boxes) * self.weight_dict.get("loss_fgl", 1.0)}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {"labels": self.loss_labels, "rboxes": self.loss_rboxes, "fgl": self.loss_fgl}
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _match(self, outputs, targets):
        matched = self.matcher(outputs, targets)
        if isinstance(matched, dict):
            return matched["indices"]
        return matched

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self._match(outputs_without_aux, targets)

        device = next(v for v in outputs.values() if torch.is_tensor(v)).device
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self._match(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                aux_indices = self._match(aux_outputs, targets)
                for loss in [x for x in self.losses if x in ("labels", "rboxes")]:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes)
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "pre_outputs" in outputs:
            pre_indices = self._match(outputs["pre_outputs"], targets)
            for loss in [x for x in self.losses if x in ("labels", "rboxes")]:
                l_dict = self.get_loss(loss, outputs["pre_outputs"], targets, pre_indices, num_boxes)
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in losses.items()}
