import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

from .box_ops import rbox_to_corners, rotated_iou 
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

            # 分类代价 (Classification Cost)
            neg_cost_class = (1 - self.alpha) * (out_prob[i*num_queries:(i+1)*num_queries] ** self.gamma) * \
                             (-(1 - out_prob[i*num_queries:(i+1)*num_queries] + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob[i*num_queries:(i+1)*num_queries]) ** self.gamma) * \
                             (-(out_prob[i*num_queries:(i+1)*num_queries] + 1e-8).log())
            
            # 使用安全标签索引
            cost_class = pos_cost_class[:, tgt_ids_safe] - neg_cost_class[:, tgt_ids_safe]

            # 顶点几何代价 (L1 Distance)
            out_corners = rbox_to_corners(out_bbox[i*num_queries:(i+1)*num_queries]) 
            tgt_corners = rbox_to_corners(tgt_bbox) 
            dist_matrix = torch.cdist(out_corners.flatten(1), tgt_corners.flatten(1), p=1) 
            cost_bbox = dist_matrix / 4.0

            # 旋转 IoU 代价
            iou_matrix = rotated_iou(out_bbox[i*num_queries:(i+1)*num_queries], tgt_bbox)
            cost_giou = -iou_matrix

            # 组合总代价
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            
            # 匈牙利算法求解
            indices.append(linear_sum_assignment(C.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]