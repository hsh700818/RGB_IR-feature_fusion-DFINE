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
    """Sanitize [cx, cy, w, h, angle] boxes for mmcv rotated IoU (eval only)."""
    rboxes = torch.nan_to_num(rboxes, nan=0.0, posinf=1.0, neginf=0.0)
    xy = rboxes[..., :2].clamp(0.0, 1.0)
    wh = rboxes[..., 2:4].clamp(min=1e-4, max=1.0)
    angle = rboxes[..., 4:5].clamp(min=-math.pi / 2, max=math.pi / 2)
    return torch.cat([xy, wh, angle], dim=-1).contiguous().float()


def rotated_iou(boxes1, boxes2, is_aligned=False):
    """True rotated IoU via mmcv — used for matching and eval metrics only.

    Kept no-grad because mmcv's CUDA kernel is not a stable gradient source
    in this setup (NaN risk after several epochs).  Training gradients come
    from gwd_loss / kld_loss instead.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        if is_aligned:
            return boxes1.new_zeros((boxes1.shape[0],))
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    try:
        from mmcv.ops import box_iou_rotated
    except Exception as exc:
        raise ImportError(
            "rotated_iou (eval) 需要 mmcv CUDA ops。"
            "请安装带 CUDA ops 的 mmcv，或将 eval_obb 设为 False。"
        ) from exc

    b1 = _prepare_rboxes_for_mmcv(boxes1.detach())
    b2 = _prepare_rboxes_for_mmcv(boxes2.detach())
    with torch.no_grad():
        out = box_iou_rotated(b1, b2, mode="iou", aligned=is_aligned, clockwise=False)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    return out.to(device=boxes1.device, dtype=boxes1.dtype)


# ---------------------------------------------------------------------------
# Differentiable rotated-box losses (GWD / KLD)
# ---------------------------------------------------------------------------

def _rbox_to_gaussian(rboxes: Tensor, eps: float = 1e-7):
    """Convert [cx, cy, w, h, angle] to (mu, Sigma) of a 2-D Gaussian.

    Each rotated box is modelled as a 2-D Gaussian whose covariance encodes
    the box shape and orientation.  This is the standard GWD/KLD formulation
    (Yang et al., CVPR 2021 / NeurIPS 2021).

    Returns:
        mu:    [..., 2]   — centre coordinates
        sigma: [..., 2, 2] — 2×2 covariance matrix (full, symmetric, PSD)
    """
    cx, cy, w, h, angle = rboxes.unbind(-1)
    w = w.clamp(min=eps)
    h = h.clamp(min=eps)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # half-axes squared
    a = (w / 2.0) ** 2
    b = (h / 2.0) ** 2

    # Sigma = R * diag(a, b) * R^T
    sigma_xx = a * cos_a ** 2 + b * sin_a ** 2
    sigma_yy = a * sin_a ** 2 + b * cos_a ** 2
    sigma_xy = (a - b) * cos_a * sin_a

    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack(
        [sigma_xx, sigma_xy, sigma_xy, sigma_yy], dim=-1
    ).view(*rboxes.shape[:-1], 2, 2)
    return mu, sigma


def gwd_loss(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7, tau: float = 1.0) -> Tensor:
    """Gaussian Wasserstein Distance loss for aligned rotated boxes.

    Fully differentiable — no external CUDA ops, no no_grad wrapper.
    Implements W2^2(G1, G2) = ||mu1 - mu2||^2 + Bures(Sigma1, Sigma2).

    The Bures metric between PSD matrices A and B is:
        tr(A) + tr(B) - 2 * tr((A^{1/2} B A^{1/2})^{1/2})

    For 2×2 matrices the matrix square root has a closed form, making this
    numerically stable and cheap.

    Args:
        boxes1, boxes2: [..., 5] tensors of (cx, cy, w, h, angle) boxes.
        tau: temperature for the normalised similarity score (default 1.0).

    Returns:
        similarity: [...] tensor in [0, 1], higher = more similar.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros(boxes1.shape[:-1])

    mu1, s1 = _rbox_to_gaussian(boxes1, eps)
    mu2, s2 = _rbox_to_gaussian(boxes2, eps)

    # Centre distance term
    d_mu = ((mu1 - mu2) ** 2).sum(dim=-1)

    # Bures metric: tr(S1) + tr(S2) - 2*tr((S1^{1/2} S2 S1^{1/2})^{1/2})
    # For 2×2 PSD matrix M, tr(M^{1/2}) = sqrt(tr(M) + 2*sqrt(det(M)))
    # We need tr((S1^{1/2} S2 S1^{1/2})^{1/2}).
    # Closed-form: let P = S1^{1/2} S2 S1^{1/2}.
    # tr(P^{1/2}) = sqrt(tr(P) + 2*sqrt(det(P)))

    tr_s1 = s1[..., 0, 0] + s1[..., 1, 1]
    tr_s2 = s2[..., 0, 0] + s2[..., 1, 1]

    det_s1 = (s1[..., 0, 0] * s1[..., 1, 1] - s1[..., 0, 1] ** 2).clamp(min=eps)
    det_s2 = (s2[..., 0, 0] * s2[..., 1, 1] - s2[..., 0, 1] ** 2).clamp(min=eps)

    # S1^{1/2} closed form for 2×2 PSD:
    #   M^{1/2} = (M + sqrt(det(M)) * I) / tr(M^{1/2})
    # where tr(M^{1/2}) = sqrt(tr(M) + 2*sqrt(det(M)))
    sqrt_det_s1 = det_s1.clamp(min=0.0).sqrt()
    tau_s1 = (tr_s1 + 2.0 * sqrt_det_s1).clamp(min=eps).sqrt()  # tr(S1^{1/2})

    s1_sqrt_xx = (s1[..., 0, 0] + sqrt_det_s1) / tau_s1
    s1_sqrt_yy = (s1[..., 1, 1] + sqrt_det_s1) / tau_s1
    s1_sqrt_xy = s1[..., 0, 1] / tau_s1

    # P = S1^{1/2} @ S2 @ S1^{1/2}  (2×2 symmetric product)
    # Row 0 of S1^{1/2} @ S2:
    r00 = s1_sqrt_xx * s2[..., 0, 0] + s1_sqrt_xy * s2[..., 1, 0]
    r01 = s1_sqrt_xx * s2[..., 0, 1] + s1_sqrt_xy * s2[..., 1, 1]
    # Row 1 of S1^{1/2} @ S2:
    r10 = s1_sqrt_xy * s2[..., 0, 0] + s1_sqrt_yy * s2[..., 1, 0]
    r11 = s1_sqrt_xy * s2[..., 0, 1] + s1_sqrt_yy * s2[..., 1, 1]

    # P = (S1^{1/2} @ S2) @ S1^{1/2}
    p00 = r00 * s1_sqrt_xx + r01 * s1_sqrt_xy
    p11 = r10 * s1_sqrt_xy + r11 * s1_sqrt_yy
    p01 = r00 * s1_sqrt_xy + r01 * s1_sqrt_yy

    tr_p = p00 + p11
    det_p = (p00 * p11 - p01 ** 2).clamp(min=0.0)
    tr_p_sqrt = (tr_p + 2.0 * det_p.sqrt()).clamp(min=0.0).sqrt()

    bures = (tr_s1 + tr_s2 - 2.0 * tr_p_sqrt).clamp(min=0.0)
    w2_sq = d_mu + bures

    # Convert distance to similarity in [0, 1]
    similarity = 1.0 / (w2_sq / tau + 1.0)
    return similarity.clamp(0.0, 1.0)


def kld_loss(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """KL-Divergence loss between Gaussian representations of rotated boxes.

    KLD(G1 || G2) + KLD(G2 || G1) (symmetrised).  Fully differentiable.
    Converted to a similarity score in [0, 1] via 1/(1 + KLD).

    Args:
        boxes1, boxes2: [..., 5] tensors of (cx, cy, w, h, angle) boxes.

    Returns:
        similarity: [...] tensor in [0, 1].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros(boxes1.shape[:-1])

    mu1, s1 = _rbox_to_gaussian(boxes1, eps)
    mu2, s2 = _rbox_to_gaussian(boxes2, eps)

    def _kld_single(mu_p, s_p, mu_q, s_q):
        # KLD(p||q) for 2-D Gaussians
        # = 0.5 * [tr(S_q^{-1} S_p) + (mu_q-mu_p)^T S_q^{-1} (mu_q-mu_p) - 2 + ln(det(S_q)/det(S_p))]
        det_p = (s_p[..., 0, 0] * s_p[..., 1, 1] - s_p[..., 0, 1] ** 2).clamp(min=eps)
        det_q = (s_q[..., 0, 0] * s_q[..., 1, 1] - s_q[..., 0, 1] ** 2).clamp(min=eps)

        # S_q^{-1} (2×2 analytic inverse)
        inv_q_xx = s_q[..., 1, 1] / det_q
        inv_q_yy = s_q[..., 0, 0] / det_q
        inv_q_xy = -s_q[..., 0, 1] / det_q

        # tr(S_q^{-1} S_p)
        tr_term = (
            inv_q_xx * s_p[..., 0, 0]
            + 2.0 * inv_q_xy * s_p[..., 0, 1]
            + inv_q_yy * s_p[..., 1, 1]
        )

        # (mu_q - mu_p)^T S_q^{-1} (mu_q - mu_p)
        d = mu_q - mu_p
        maha = (
            inv_q_xx * d[..., 0] ** 2
            + 2.0 * inv_q_xy * d[..., 0] * d[..., 1]
            + inv_q_yy * d[..., 1] ** 2
        )

        log_ratio = (det_q / det_p).clamp(min=eps).log()
        return 0.5 * (tr_term + maha - 2.0 + log_ratio)

    kld_12 = _kld_single(mu1, s1, mu2, s2).clamp(min=0.0)
    kld_21 = _kld_single(mu2, s2, mu1, s1).clamp(min=0.0)
    sym_kld = (kld_12 + kld_21) * 0.5

    similarity = 1.0 / (sym_kld + 1.0)
    return similarity.clamp(0.0, 1.0)


def differentiable_rotated_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-6) -> Tensor:
    """Legacy AABB-IoU × angle-cosine surrogate — kept for backward compat.

    Prefer gwd_loss() or kld_loss() for training; this is retained so that
    configs with loss_riou still work without changes.
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
