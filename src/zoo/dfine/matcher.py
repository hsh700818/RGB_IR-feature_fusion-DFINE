import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from .box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    rbox_to_corners,
    rotated_iou,
)
from ...core import register

__all__ = ['HungarianMatcher']


@register()
class HungarianMatcher(nn.Module):
    __inject__ = []

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 weight_dict: dict = None,
                 **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # 从 weight_dict 获取匹配权重，若无则使用默认
        if weight_dict is None:
            weight_dict = {'cost_class': 2.0, 'cost_bbox': 5.0, 'cost_giou': 2.0}
        self.cost_class = weight_dict.get('cost_class', 2.0)
        self.cost_bbox = weight_dict.get('cost_bbox', 5.0)
        self.cost_giou = weight_dict.get('cost_giou', 2.0)

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        num_classes = outputs["pred_logits"].shape[-1]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        is_hbb = out_bbox.shape[-1] == 4

        indices = []
        for i in range(bs):
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]

            if len(tgt_ids) == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            # 防止类别 ID 越界导致 CUDA 报错
            # 如果 tgt_ids 包含 >= num_classes 的值，计算 cost_class 时会崩溃
            tgt_ids_safe = torch.clamp(tgt_ids, 0, num_classes - 1)

            prob_i = out_prob[i * num_queries:(i + 1) * num_queries]
            bbox_i = out_bbox[i * num_queries:(i + 1) * num_queries]

            # 分类代价 (Classification Cost)
            neg_cost_class = (1 - self.alpha) * (prob_i ** self.gamma) * \
                             (-(1 - prob_i + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - prob_i) ** self.gamma) * \
                             (-(prob_i + 1e-8).log())

            # 使用安全标签索引
            cost_class = pos_cost_class[:, tgt_ids_safe] - neg_cost_class[:, tgt_ids_safe]

            if is_hbb:
                # HBB 消融分支：预测框为 [cx, cy, w, h]，标注若为 5 维则取前 4 维。
                tgt_bbox_hbb = tgt_bbox[..., :4]
                cost_bbox = torch.cdist(bbox_i, tgt_bbox_hbb, p=1)

                out_xyxy = box_cxcywh_to_xyxy(bbox_i)
                tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox_hbb)
                giou_matrix = generalized_box_iou(out_xyxy, tgt_xyxy)
                cost_giou = -giou_matrix
            else:
                # OBB 分支：使用顶点几何代价和旋转 IoU 代价。
                out_corners = rbox_to_corners(bbox_i)
                tgt_corners = rbox_to_corners(tgt_bbox)
                dist_matrix = torch.cdist(out_corners.flatten(1), tgt_corners.flatten(1), p=1)
                cost_bbox = dist_matrix / 4.0

                iou_matrix = rotated_iou(bbox_i, tgt_bbox)
                cost_giou = -iou_matrix

            # 组合总代价
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            # 匈牙利算法求解
            indices.append(linear_sum_assignment(C.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
