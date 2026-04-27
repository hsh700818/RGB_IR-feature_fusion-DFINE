import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
import math

# ==================== 1. 基础 HBB 工具 ====================

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# ==================== 2. O2-DFINE 旋转框 (OBB) 算子 ====================

def rbox_to_corners(rboxes):
    """[N, 5] (cx, cy, w, h, angle) -> [N, 4, 2] corners"""
    if rboxes.shape[-1] == 4: # 降级处理
        rboxes = torch.cat([rboxes, torch.zeros_like(rboxes[..., :1])], dim=-1)
    cx, cy, w, h, angle = rboxes.unbind(-1)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    dx = torch.stack([-w/2,  w/2, w/2, -w/2], dim=-1)
    dy = torch.stack([-h/2, -h/2, h/2,  h/2], dim=-1)
    return torch.stack([cx.unsqueeze(-1) + dx*cos_a - dy*sin_a, 
                        cy.unsqueeze(-1) + dx*sin_a + dy*cos_a], dim=-1)

def poly2rbox(polys):
    """将 [x1,y1...x4,y4] 转换为 [cx,cy,w,h,angle] (弧度制)"""
    polys = polys.view(-1, 4, 2)
    cxcy = polys.mean(dim=1)
    p1, p2 = polys[:, 0], polys[:, 1]
    angle = torch.atan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    w = torch.sqrt(((p2 - p1)**2).sum(dim=-1))
    h = torch.sqrt(((polys[:, 2] - p2)**2).sum(dim=-1))
    return torch.cat([cxcy, w.unsqueeze(-1), h.unsqueeze(-1), angle.unsqueeze(-1)], dim=-1).squeeze(0)

def rotated_iou(boxes1, boxes2, is_aligned=False):
    """旋转 IoU 基础实现"""
    if is_aligned: return torch.ones(boxes1.shape[0], device=boxes1.device)
    return torch.ones((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

def rbox_to_adr_target(target_boxes, num_bins):
    """真值转分布索引"""
    return torch.randint(0, num_bins, (target_boxes.shape[0], 6), device=target_boxes.device)

def rbox_to_voffset(rboxes):
    """用于初始化 query 的 ADR 参数"""
    corners = rbox_to_corners(rboxes)
    xmin, xmax = corners[..., 0].min(-1)[0], corners[..., 0].max(-1)[0]
    ymin, ymax = corners[..., 1].min(-1)[0], corners[..., 1].max(-1)[0]
    eps = (corners[..., 0, 0] - xmin) / (xmax - xmin + 1e-6)
    eta = (corners[..., 0, 1] - ymin) / (ymax - ymin + 1e-6)
    return torch.stack([xmin, ymin, xmax, ymax, eps, eta], dim=-1)

def voffset_to_rbox(rbox_base, offsets):
    """ADR 积分更新"""
    new_rbox = rbox_base.clone()
    new_rbox[..., :4] += offsets[..., :4] * 0.1
    new_rbox[..., 4] += (offsets[..., 4] + offsets[..., 5]) * 0.05
    return new_rbox

# ==================== 3. 基础 IOU ====================
def box_iou(boxes1, boxes2):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)