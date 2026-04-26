import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ...core import register
from .utils import MLP, inverse_sigmoid
from .rotated_box_ops import rbox_to_voffset, voffset_to_rbox # 需确保存在此算子库

__all__ = ['DFINETransformerDecoder']

class O2DFINEHead(nn.Module):
    def __init__(self, hidden_dim, num_bins):
        super().__init__()
        # O2-DFINE 核心：预测 6 个分布 (4个外部矩形边 + 2个顶点偏移)
        self.num_bins = num_bins
        self.reg_conf = nn.Linear(hidden_dim, 6 * num_bins)

    def forward(self, x):
        # x: [L, B, 300, hidden_dim]
        # output: [L, B, 300, 6, num_bins]
        out = self.reg_conf(x)
        return out.view(out.shape[:-1] + (6, self.num_bins))

@register()
class DFINETransformerDecoder(nn.Module):
    __inject__ = ['num_layers', 'hidden_dim', 'num_bins']

    def __init__(self, 
                 num_layers=6, 
                 hidden_dim=256, 
                 num_bins=16, 
                 num_classes=5,
                 eval_idx=-1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.eval_idx = eval_idx

        # 1. 角度分布精细化 (ADR) 回归头
        self.reg_head = nn.ModuleList([
            O2DFINEHead(hidden_dim, num_bins) for _ in range(num_layers)
        ])
        
        # 2. 分类头
        self.cls_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])

        # 3. ADR 权重映射函数 A(n) (参考 O2-DFINE 论文公式 4)
        self.register_buffer('adr_weights', self._gen_adr_weights(num_bins))

    def _gen_adr_weights(self, N):
        """生成论文中的 A(n) 权重张量"""
        n = torch.arange(N)
        mid = (N - 1) / 2
        # 控制曲线形状的超参数 a, c
        a, c = 2.0, 1.0 
        sgn = torch.sign(n - mid)
        # 实现论文中的分段函数逻辑
        weights = torch.where(
            (n == 0) | (n == N - 1),
            torch.tensor(2 * a),
            c * ((1 + a/c)**(2 * torch.abs(n - mid) / (N - 2)) - 1)
        )
        return sgn * weights

    def forward(self, refinement_features, reference_points, memory, spatial_shapes, level_start_index):
        """
        Args:
            refinement_features: Encoder 输出的特征
            reference_points: 初始参考框 [B, 300, 5] (cx, cy, w, h, angle)
            memory: 图像全局记忆
        """
        output_classes = []
        output_distributions = []
        
        # 初始预测 (Layer 0)
        curr_rbox = reference_points 
        
        # 逐层迭代精细化
        for i in range(self.num_layers):
            # 1. 旋转 Cross-Attention (Sampling points rotated by theta)
            # 这里需调用带旋转角度的变形注意力机制
            hs = self.layers[i](
                refinement_features, 
                curr_rbox, # 传入带角度的 rbox 进行旋转采样
                memory, 
                spatial_shapes, 
                level_start_index
            )

            # 2. 预测残差分布 logits (ADR)
            # logits 包含 4个边 + 2个 offset 的分布
            logits = self.reg_head[i](hs) # [B, 300, 6, num_bins]
            
            # 3. 分布积分获得偏移量 (Integral Regression)
            probs = F.softmax(logits, dim=-1)
            # offset = sum(A(n) * P(n))
            offsets = torch.sum(probs * self.adr_weights, dim=-1) # [B, 300, 6]

            # 4. 更新旋转框 (Residual Update)
            # 逻辑：外部矩形 (cx, cy, Wr, Hr) + 顶点偏移 (eps, eta) -> 新的 (cx, cy, w, h, theta)
            curr_rbox = self.apply_adr_update(curr_rbox, offsets)

            # 5. 分类预测
            class_logits = self.cls_head[i](hs)

            output_classes.append(class_logits)
            output_distributions.append(logits)

        # 返回各层结果用于训练和推理
        out = {'pred_logits': output_classes, 'pred_reg_logits': output_distributions}
        return out

    def apply_adr_update(self, rbox, offsets):
        """根据 ADR 预测的 6 个偏移量更新旋转框"""
        # 1. 拆分 rbox 关键点
        # 2. 将 offsets 作用于外部矩形和顶点偏移
        # 3. 将 (ExternalRect, VOffset) 转回 (cx, cy, w, h, theta)
        # 此处封装了复杂的三角几何转换
        new_rbox = voffset_to_rbox(rbox, offsets) 
        return new_rbox.detach() # 梯度通过分布 logits 传播