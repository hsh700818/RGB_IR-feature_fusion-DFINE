import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sigmoid_focal_loss
from .rotated_box_ops import rbox_to_corners, rotated_iou # 需确保存在旋转框算子库
from ...core import register

__all__ = ['DFINECriterion']

@register()
class DFINECriterion(nn.Module):
    __inject__ = ['matcher', 'num_classes', 'weight_dict', 'losses', 'num_bins']

    def __init__(self, matcher, num_classes, weight_dict, losses, num_bins):
        """
        Args:
            matcher: HungarianMatcher 实例
            num_classes: 类别数 (DroneVehicle 为 5)
            weight_dict: 损失权重字典
            losses: 损失类型列表 (如 ['labels', 'rboxes', 'fgl'])
            num_bins: ADR 分布的 bin 数量 (通常为 16)
        """
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_bins = num_bins

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """分类损失 (使用 Varifocal Loss 或 Focal Loss)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = sigmoid_focal_loss(src_logits, target_classes, self.num_classes, num_boxes=num_boxes, alpha=0.25, gamma=2.0)
        losses = {'loss_vfl': loss_ce}
        return losses

    def loss_rboxes(self, outputs, targets, indices, num_boxes):
        """旋转框回归损失 (Chamfer Distance + Rotated IoU)"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] # [num_matched_queries, 5] (cx, cy, w, h, angle)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            return {'loss_bbox': src_boxes.sum() * 0, 'loss_riou': src_boxes.sum() * 0}

        # 1. 旋转框顶点 Chamfer 损失 (基于顶点的几何对齐)
        src_corners = rbox_to_corners(src_boxes) # [N, 4, 2]
        target_corners = rbox_to_corners(target_boxes) # [N, 4, 2]
        loss_bbox = F.l1_loss(src_corners, target_corners, reduction='none').sum() / num_boxes

        # 2. 旋转 IoU 损失 (针对 DroneVehicle 密集小目标的重叠惩罚)
        # 注意：这里调用自定义的 rotated_iou 算子
        iou_vector = rotated_iou(src_boxes, target_boxes, is_aligned=True)
        loss_riou = (1 - iou_vector).sum() / num_boxes

        losses = {}
        losses['loss_bbox'] = loss_bbox
        losses['loss_riou'] = loss_riou
        return losses

    def loss_fgl(self, outputs, targets, indices, num_boxes):
        """角度分布精细化 (ADR) 损失 - 对应论文 Sec IV-B"""
        assert 'pred_reg_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # 这里的 pred_reg_logits 是 ADR 输出的 6 个分布的分布情况
        # [B, 300, 6, num_bins]
        src_reg_logits = outputs['pred_reg_logits'][idx] 
        
        # 准备真值分布目标 (由 GT 旋转框解算出的 6 个分布位置)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # 将 target_boxes (cx, cy, w, h, theta) 转换为 ADR 所需的 (edges_4 + offset_2)
        # 这里需要调用特殊的算子转换
        from .rotated_box_ops import rbox_to_adr_target
        target_adr = rbox_to_adr_target(target_boxes, self.num_bins) # [N, 6]

        # 计算分布损失 (DFL / CrossEntropy)
        # 惩罚预测分布偏离真值 bin 的程度
        loss_fgl = F.cross_entropy(src_reg_logits.flatten(0, 1), 
                                   target_adr.flatten().long(), 
                                   reduction='none').view(src_reg_logits.shape[:2])
        
        losses = {'loss_fgl': loss_fgl.sum() / num_boxes}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'rboxes': self.loss_rboxes,
            'fgl': self.loss_fgl,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """调用此函数进行整个 Batch 的 Loss 计算"""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # 1. 匈牙利匹配 (使用 Chamfer Distance Matcher)
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 计算匹配到的框数量
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.cuda.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1, min=1).item()

        # 3. 计算所有定义的 Losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 4. 计算辅助层损失 (Auxiliary Losses)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses