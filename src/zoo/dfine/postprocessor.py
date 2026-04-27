"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def _obb_to_hbb_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized 5D OBB [cx, cy, w, h, angle] to normalized HBB xyxy."""
        from .box_ops import rbox_to_corners

        corners = rbox_to_corners(boxes)
        xmin = corners[..., 0].min(dim=-1).values
        ymin = corners[..., 1].min(dim=-1).values
        xmax = corners[..., 0].max(dim=-1).values
        ymax = corners[..., 1].max(dim=-1).values
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1).clamp(0, 1)

    def _boxes_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.shape[-1] == 5:
            return self._obb_to_hbb_xyxy(boxes)
        if boxes.shape[-1] == 4:
            return torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").clamp(0, 1)
        raise ValueError(f"Unsupported pred_boxes shape {boxes.shape}; expected last dim 4 or 5")

    def _scale_rboxes(self, boxes: torch.Tensor, orig_target_sizes: torch.Tensor):
        """Scale normalized OBB to image pixels while preserving angle radians."""
        if boxes.shape[-1] != 5:
            return None
        scale = torch.cat(
            [orig_target_sizes, orig_target_sizes, torch.ones_like(orig_target_sizes[:, :1])], dim=1
        )
        # orig_target_sizes is [W, H], so scale is [W, H, W, H, 1].
        return boxes * scale.unsqueeze(1)

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]

        hbb_pred = self._boxes_to_xyxy(pred_boxes)
        hbb_pred = hbb_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        rbox_pred = self._scale_rboxes(pred_boxes, orig_target_sizes)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = hbb_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, hbb_pred.shape[-1])
            )
            rboxes = None
            if rbox_pred is not None:
                rboxes = rbox_pred.gather(
                    dim=1, index=index.unsqueeze(-1).repeat(1, 1, rbox_pred.shape[-1])
                )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = hbb_pred
            rboxes = rbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                if rboxes is not None:
                    rboxes = torch.gather(
                        rboxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, rboxes.shape[-1])
                    )

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
            result = dict(labels=lab, boxes=box, scores=sco)
            if rboxes is not None:
                result["rboxes"] = rboxes[i]
            results.append(result)

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
