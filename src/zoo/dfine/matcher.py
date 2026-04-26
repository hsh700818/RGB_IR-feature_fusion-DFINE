import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

from .rotated_box_ops import rbox_to_corners, rotated_iou # 需确保存在旋转框算子库
from ...core import register

__all__ = ['HungarianMatcher']

@register()
class HungarianMatcher(nn.Module):
    __inject__ = ['cost_class', 'cost_bbox', 'cost_giou']

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """
        Args:
            cost_class: 分类损失权重
            cost_bbox:  旋转框几何距离 (Chamfer/L1) 权重
            cost_giou:  旋转 IoU 权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        计算预测与真值之间的 Cost Matrix 并进行最优分配
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 计算分类 Cost (使用 Softmax 概率)
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # [batch_size * num_queries, num_classes]
        
        # 2. 准备预测框和真值框
        # outputs["pred_boxes"] 维度应为 [batch_size, num_queries, 5] (cx, cy, w, h, angle)
        out_bbox = outputs["pred_boxes"].flatten(0, 1) 

        indices = []
        for i in range(bs):
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"] # [num_gts, 5]

            if len(tgt_ids) == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            # --- A. 分类 Cost ---
            # 使用 Focal Loss 的启发式 Cost
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob[i*num_queries:(i+1)*num_queries] ** gamma) * \
                             (-(1 - out_prob[i*num_queries:(i+1)*num_queries] + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob[i*num_queries:(i+1)*num_queries]) ** gamma) * \
                             (-(out_prob[i*num_queries:(i+1)*num_queries] + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # --- B. 旋转框几何 Cost (参考 O2-DFINE Chamfer Distance) ---
            # 逻辑：计算两个框顶点之间的 L1 距离之和
            # out_bbox: [num_queries, 5], tgt_bbox: [num_gts, 5]
            out_corners = rbox_to_corners(out_bbox[i*num_queries:(i+1)*num_queries]) # [300, 4, 2]
            tgt_corners = rbox_to_corners(tgt_bbox) # [num_gts, 4, 2]
            
            # 计算倒角距离 (Chamfer-like L1 cost on vertices)
            # 对每一个预测框，计算其 4 个点到真值框 4 个点的最小对应距离
            dist_matrix = torch.cdist(out_corners.flatten(1), tgt_corners.flatten(1), p=1) # [300, num_gts]
            cost_bbox = dist_matrix / 4.0 # 均值化

            # --- C. 旋转 IoU Cost ---
            # 旋转 IoU 计算复杂度高，但在匹配阶段能有效排除错位框
            # 返回 [300, num_gts] 的 IoU 矩阵
            iou_matrix = rotated_iou(out_bbox[i*num_queries:(i+1)*num_queries], tgt_bbox)
            cost_giou = -iou_matrix # IoU 越大，代价越小

            # 最终 Cost Matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()

            # Hungarian Algorithm 执行最优分配
            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]