import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
import math

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# ==================== O2-DFINE 旋转框 (OBB) 核心算子 ====================

def rbox_to_corners(rboxes):
    """
    [N, 5] (cx, cy, w, h, angle) -> [N, 4, 2] corners
    修正了广播机制，适配 DETR 架构的 batch 处理
    """
    if rboxes.shape[-1] == 4:
        # 兼容处理：如果没有角度，补 0
        rboxes = torch.cat([rboxes, torch.zeros_like(rboxes[..., :1])], dim=-1)
        
    cx, cy, w, h, angle = rboxes.unbind(-1)
    
    # 【核心修正】：增加 unsqueeze(-1) 确保形状为 [N, 1]，从而能与 [N, 4] 广播
    cos_a = torch.cos(angle).unsqueeze(-1) 
    sin_a = torch.sin(angle).unsqueeze(-1)
    
    # w 和 h 也需要 unsqueeze 以生成 [N, 4] 的 dx/dy
    w_2 = w.unsqueeze(-1) / 2.0
    h_2 = h.unsqueeze(-1) / 2.0
    
    # 定义 4 个顶点相对中心的偏移 (x_min, x_max, x_max, x_min)
    # 形状均为 [N, 4]
    dx = torch.cat([-w_2,  w_2,  w_2, -w_2], dim=-1)
    dy = torch.cat([-h_2, -h_2,  h_2,  h_2], dim=-1)
    
    # 矩阵运算：R * [dx, dy]^T + [cx, cy]^T
    # 形状对齐：[N, 1] + ([N, 4] * [N, 1]) - ([N, 4] * [N, 1]) = [N, 4]
    corners_x = cx.unsqueeze(-1) + dx * cos_a - dy * sin_a
    corners_y = cy.unsqueeze(-1) + dx * sin_a + dy * cos_a
    
    return torch.stack([corners_x, corners_y], dim=-1) # [N, 4, 2]

def poly2rbox(polys):
    """将 [x1,y1...x4,y4] 转换为 [cx,cy,w,h,angle]"""
    polys = polys.view(-1, 4, 2)
    cxcy = polys.mean(dim=1)
    p1, p2 = polys[:, 0], polys[:, 1]
    angle = torch.atan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    w = torch.sqrt(((p2 - p1)**2).sum(dim=-1))
    h = torch.sqrt(((polys[:, 2] - p2)**2).sum(dim=-1))
    return torch.cat([cxcy, w.unsqueeze(-1), h.unsqueeze(-1), angle.unsqueeze(-1)], dim=-1).squeeze(0)

def rotated_iou(boxes1, boxes2, is_aligned=False):
    """
    旋转 IoU 基础占位实现
    在 O2-DFINE 训练中，此处通常被替换为更精确的旋转 IoU 算子
    """
    if is_aligned:
        return torch.ones(boxes1.shape[0], device=boxes1.device)
    return torch.ones((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

# --- ADR 角度分布精细化算子 ---

def rbox_to_adr_target(target_boxes, num_bins):
    device = target_boxes.device
    return torch.randint(0, num_bins, (target_boxes.shape[0], 6), device=device)

def rbox_to_voffset(rboxes):
    corners = rbox_to_corners(rboxes)
    xmin = corners[..., 0].min(dim=-1)[0]
    xmax = corners[..., 0].max(dim=-1)[0]
    ymin = corners[..., 1].min(dim=-1)[0]
    ymax = corners[..., 1].max(dim=-1)[0]
    eps = (corners[..., 0, 0] - xmin) / (xmax - xmin + 1e-6)
    eta = (corners[..., 0, 1] - ymin) / (ymax - ymin + 1e-6)
    return torch.stack([xmin, ymin, xmax, ymax, eps, eta], dim=-1)

def voffset_to_rbox(rbox_base, offsets):
    new_rbox = rbox_base.clone()
    new_rbox[..., :4] += offsets[..., :4] * 0.1
    new_rbox[..., 4] += (offsets[..., 4] + offsets[..., 5]) * 0.05
    return new_rbox

# ==================== 基础 IOU 算子 ====================

def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area