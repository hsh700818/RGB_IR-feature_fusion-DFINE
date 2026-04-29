import torch
from torch import Tensor
from torchvision.ops.boxes import box_area
import math


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# ==================== O2-DFINE 旋转框 (OBB) 核心算子 ====================

def rbox_to_corners(rboxes):
    """
    [N, 5] (cx, cy, w, h, angle) -> [N, 4, 2] corners.
    Also supports tensors with leading dimensions, e.g. [B, Q, 5].
    """
    if rboxes.shape[-1] == 4:
        rboxes = torch.cat([rboxes, torch.zeros_like(rboxes[..., :1])], dim=-1)

    cx, cy, w, h, angle = rboxes.unbind(-1)
    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)

    w_2 = w.clamp(min=0.0).unsqueeze(-1) / 2.0
    h_2 = h.clamp(min=0.0).unsqueeze(-1) / 2.0

    dx = torch.cat([-w_2, w_2, w_2, -w_2], dim=-1)
    dy = torch.cat([-h_2, -h_2, h_2, h_2], dim=-1)

    corners_x = cx.unsqueeze(-1) + dx * cos_a - dy * sin_a
    corners_y = cy.unsqueeze(-1) + dx * sin_a + dy * cos_a
    return torch.stack([corners_x, corners_y], dim=-1)


def poly2rbox(polys):
    """将 [x1,y1...x4,y4] 转换为 [cx,cy,w,h,angle]"""
    polys = polys.view(-1, 4, 2)
    cxcy = polys.mean(dim=1)
    p1, p2 = polys[:, 0], polys[:, 1]
    angle = torch.atan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    w = torch.sqrt(((p2 - p1) ** 2).sum(dim=-1))
    h = torch.sqrt(((polys[:, 2] - p2) ** 2).sum(dim=-1))
    return torch.cat([cxcy, w.unsqueeze(-1), h.unsqueeze(-1), angle.unsqueeze(-1)], dim=-1).squeeze(0)


def _prepare_rboxes_for_mmcv(rboxes: Tensor) -> Tensor:
    """Prepare valid [cx, cy, w, h, angle] boxes for mmcv rotated IoU.

    mmcv rotated IoU kernels are sensitive to NaN/Inf and near-zero widths or
    heights.  Sanitizing here prevents invalid values from entering the CUDA op.
    """
    rboxes = torch.nan_to_num(rboxes, nan=0.0, posinf=1.0, neginf=0.0)
    xy = rboxes[..., :2].clamp(0.0, 1.0)
    wh = rboxes[..., 2:4].clamp(min=1e-4, max=1.0)
    angle = rboxes[..., 4:5].clamp(min=-math.pi / 2, max=math.pi / 2)
    return torch.cat([xy, wh, angle], dim=-1).contiguous().float()


def rotated_iou(boxes1, boxes2, is_aligned=False):
    """True rotated IoU using mmcv.ops.box_iou_rotated.

    Important: this intentionally avoids mmcv diff_iou_rotated during training.
    In the current environment diff_iou_rotated can produce unstable gradients
    and make decoder boxes become NaN after several epochs.  box_iou_rotated
    still computes true polygon rotated IoU, but its output is treated as a
    stable IoU target/metric term.  The main localization gradient continues to
    come from the corner L1 loss.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        if is_aligned:
            return boxes1.new_zeros((boxes1.shape[0],))
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    try:
        from mmcv.ops import box_iou_rotated
    except Exception as exc:
        raise ImportError(
            "loss_riou 已启用，但当前环境没有可用的 mmcv.ops.box_iou_rotated。"
            "请检查 mmcv 是否安装了 CUDA ops，或临时把 loss_riou 改回 0.0。"
        ) from exc

    b1 = _prepare_rboxes_for_mmcv(boxes1.detach())
    b2 = _prepare_rboxes_for_mmcv(boxes2.detach())
    with torch.no_grad():
        out = box_iou_rotated(b1, b2, mode="iou", aligned=is_aligned, clockwise=False)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    return out.to(device=boxes1.device, dtype=boxes1.dtype)


def differentiable_rotated_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-6) -> Tensor:
    """Differentiable IoU-like similarity for aligned rotated boxes.

    mmcv's `box_iou_rotated` is used for matching/eval but not suitable as a
    stable training gradient source in this project setup. Here we build a
    smooth surrogate from:
    1) enclosing AABB IoU computed from rotated corners
    2) angle consistency term cos(|delta angle|)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    c1 = rbox_to_corners(boxes1)
    c2 = rbox_to_corners(boxes2)

    x1_min = c1[..., 0].min(dim=-1).values
    y1_min = c1[..., 1].min(dim=-1).values
    x1_max = c1[..., 0].max(dim=-1).values
    y1_max = c1[..., 1].max(dim=-1).values

    x2_min = c2[..., 0].min(dim=-1).values
    y2_min = c2[..., 1].min(dim=-1).values
    x2_max = c2[..., 0].max(dim=-1).values
    y2_max = c2[..., 1].max(dim=-1).values

    inter_w = (torch.minimum(x1_max, x2_max) - torch.maximum(x1_min, x2_min)).clamp(min=0.0)
    inter_h = (torch.minimum(y1_max, y2_max) - torch.maximum(y1_min, y2_min)).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = (x1_max - x1_min).clamp(min=0.0) * (y1_max - y1_min).clamp(min=0.0)
    area2 = (x2_max - x2_min).clamp(min=0.0) * (y2_max - y2_min).clamp(min=0.0)
    union = (area1 + area2 - inter).clamp(min=eps)
    aabb_iou = inter / union

    dtheta = boxes1[..., 4] - boxes2[..., 4]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    angle_sim = torch.cos(dtheta).clamp(min=0.0)

    return (aabb_iou * angle_sim).clamp(0.0, 1.0)


# --- ADR 角度分布精细化算子 ---

def rbox_to_adr_target(target_boxes, num_bins):
    if target_boxes.numel() == 0:
        return target_boxes.new_zeros((0, 6), dtype=torch.long)

    voffset = rbox_to_voffset(target_boxes)
    voffset = torch.nan_to_num(voffset, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    bins = torch.floor(voffset * float(num_bins)).long()
    return bins.clamp(min=0, max=num_bins - 1)


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
