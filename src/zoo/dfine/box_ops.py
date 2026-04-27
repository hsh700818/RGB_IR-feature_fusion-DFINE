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


def _rbox_to_enclosing_xyxy(rboxes: Tensor) -> Tensor:
    corners = rbox_to_corners(rboxes)
    xmin = corners[..., 0].min(dim=-1).values
    ymin = corners[..., 1].min(dim=-1).values
    xmax = corners[..., 0].max(dim=-1).values
    ymax = corners[..., 1].max(dim=-1).values
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def _pairwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _hbb_iou_fallback(boxes1, boxes2, is_aligned=False):
    """Only used when loss_riou is disabled or for emergency debugging."""
    hbb1 = _rbox_to_enclosing_xyxy(boxes1)
    hbb2 = _rbox_to_enclosing_xyxy(boxes2)
    if is_aligned:
        lt = torch.maximum(hbb1[:, :2], hbb2[:, :2])
        rb = torch.minimum(hbb1[:, 2:], hbb2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (hbb1[:, 2] - hbb1[:, 0]).clamp(min=0) * (hbb1[:, 3] - hbb1[:, 1]).clamp(min=0)
        area2 = (hbb2[:, 2] - hbb2[:, 0]).clamp(min=0) * (hbb2[:, 3] - hbb2[:, 1]).clamp(min=0)
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-6)
    return _pairwise_box_iou(hbb1, hbb2)


def _mmcv_box_iou_rotated(boxes1, boxes2, is_aligned=False):
    from mmcv.ops import box_iou_rotated

    boxes1 = boxes1.contiguous().float()
    boxes2 = boxes2.contiguous().float()
    out = box_iou_rotated(boxes1, boxes2, mode="iou", aligned=is_aligned, clockwise=False)
    return out.to(dtype=boxes1.dtype)


def _mmcv_diff_iou_rotated_aligned(boxes1, boxes2):
    """Use MMCV differentiable rotated IoU when available.

    MMCV versions expose this operator with slightly different names.  We try
    the common names first.  The aligned loss only needs IoU for matched pairs.
    """
    try:
        from mmcv.ops import diff_iou_rotated_2d
    except Exception:
        try:
            from mmcv.ops import diff_iou_rotated as diff_iou_rotated_2d
        except Exception:
            return None

    boxes1_f = boxes1.contiguous().float()
    boxes2_f = boxes2.contiguous().float()
    out = diff_iou_rotated_2d(boxes1_f.unsqueeze(0), boxes2_f.unsqueeze(0))
    out = out.squeeze(0)

    if out.ndim == 2:
        # Some versions return [N, N].  For aligned matched pairs use diagonal.
        out = out.diag()
    elif out.ndim > 1:
        out = out.reshape(-1)
    return out.to(dtype=boxes1.dtype)


def rotated_iou(boxes1, boxes2, is_aligned=False):
    """True rotated IoU for OBB training.

    Priority:
    1. For aligned matched pairs, use MMCV differentiable rotated IoU when the
       installed mmcv provides diff_iou_rotated_2d / diff_iou_rotated.
    2. Otherwise use MMCV box_iou_rotated, which computes true rotated IoU.

    If mmcv is not installed, this function raises a clear error instead of
    silently falling back to HBB IoU, because loss_riou should be a real OBB loss.
    Temporarily set loss_riou: 0.0 if you need to train without mmcv.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        if is_aligned:
            return boxes1.new_zeros((boxes1.shape[0],))
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    if is_aligned:
        diff_iou = _mmcv_diff_iou_rotated_aligned(boxes1, boxes2)
        if diff_iou is not None:
            return diff_iou.clamp(min=0.0, max=1.0)

    try:
        return _mmcv_box_iou_rotated(boxes1, boxes2, is_aligned=is_aligned).clamp(min=0.0, max=1.0)
    except Exception as exc:
        raise ImportError(
            "loss_riou 已启用，但当前环境没有可用的 mmcv 旋转 IoU 算子。"
            "请安装与 PyTorch/CUDA 匹配的 mmcv，或临时把配置中的 loss_riou 改回 0.0。"
        ) from exc


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
