import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==================== 1. 激活函数获取工具  ====================

def get_activation(act: str, inplace: bool = True):
    """
    根据名称获取激活函数层
    """
    if act is None:
        return nn.Identity()
    
    act_lower = act.lower()
    if act_lower == 'silu':
        return nn.SiLU(inplace=inplace)
    elif act_lower == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_lower == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act_lower == 'gelu':
        return nn.GELU()
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Activation '{act}' is not implemented.")

# ==================== 2. 损失函数工具 ====================

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0,
                       class_weights=None):
    """
    DETR 类模型常用的 Focal Loss 算子。

    支持输入形状：
    - [N, C]
    - [B, Q, C]

    class_weights: optional [C] tensor of per-class multipliers.  Applied after
    the focal modulation so it only re-weights the class dimension without
    disturbing the easy/hard sample balance.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if class_weights is not None:
        # class_weights: [C] — broadcast over batch/query dims
        loss = loss * class_weights.to(loss.device)

    if loss.ndim < 2:
        return loss.sum() / num_boxes
    return loss.mean(dim=-1).sum() / num_boxes

# ==================== 3. 几何变换工具 ====================

def inverse_sigmoid(x, eps=1e-5):
    """反 Sigmoid 函数，用于坐标解算"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

# ==================== 4. 网络架构工具 ====================

class MLP(nn.Module):
    """多层感知机工具"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# ==================== 5. O2-DFINE 旋转框专用 (ADR 权重) ====================

def get_adr_weights(N, device):
    """
    生成 O2-DFINE 论文中的 A(n) 权重映射张量
    """
    n = torch.arange(N, device=device)
    mid = (N - 1) / 2
    a, c = 2.0, 1.0 
    sgn = torch.sign(n - mid)
    
    weights = torch.where(
        (n == 0) | (n == N - 1),
        torch.tensor(2 * a, device=device),
        c * ((1 + a/c)**(2 * torch.abs(n - mid) / (N - 2)) - 1)
    )
    return sgn * weights