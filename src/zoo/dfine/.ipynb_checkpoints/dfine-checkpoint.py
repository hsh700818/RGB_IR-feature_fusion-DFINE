import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ...core import register

__all__ = ['DFINE']

# 1. 全局-局部协同注意力模块 (参考 Aerial Image TGRS 2025)
# 作用：通过全局池化锁定车辆与环境的联系，利用小卷积核保护航拍小目标的边缘细节
class GLSA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), groups=ch)
        self.conv_w = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch)
        
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        # 提取垂直与水平方向的空间注意力
        x_h = self.conv_h(self.avg_pool_h(x))
        x_w = self.conv_w(self.avg_pool_w(x))
        # 提取通道权重
        x_c = self.channel_att(x)
        # 空间与通道联合增强
        return identity * x_h.sigmoid() * x_w.sigmoid() * x_c

# 2. 光照感知自适应融合模块 (参考 IMHFNet TGRS 2025)
# 作用：根据 RGB 亮度自动调整。白天光照充足时平衡两分支，全黑夜间自动抑制噪声巨大的可见光分支。
class IlluminationAwareFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # 用于提取光照强度的轻量级预测器
        self.illum_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1),
            nn.Sigmoid()
        )
        # 多模态特征融合层
        self.fusion_conv = nn.Conv2d(ch * 2, ch, 1)
        # 融合后的特征增强
        self.glsa = GLSA(ch)

    def forward(self, f_rgb, f_ir):
        # 1. 估计当前区域的光照得分 (0:深夜, 1:明亮白天)
        illum_score = self.illum_extract(f_rgb) 
        
        # 2. 动态模态调节：夜间增加 IR 占比，抑制 RGB 噪声
        # 融合公式：f_weighted = [RGB * illum, IR * (2.0 - illum)]
        f_cat = torch.cat([f_rgb * illum_score, f_ir * (2.0 - illum_score)], dim=1)
        f_fused = self.fusion_conv(f_cat)
        
        # 3. 进阶特征增强 (针对 DroneVehicle 的非对齐和微小目标)
        return self.glsa(f_fused)

@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        # 深拷贝一个相同架构的 Backbone 用于红外 (IR) 流
        self.backbone_ir = copy.deepcopy(backbone)
        self.encoder = encoder
        self.decoder = decoder
        
        # 动态获取编码器需要的通道数 (支持 P3, P4, P5 尺度)
        in_channels = encoder.in_channels 
        
        # 为每一层特征图建立光照感知融合层
        self.fusion_layers = nn.ModuleList([
            IlluminationAwareFusion(ch) for ch in in_channels
        ])

    def forward(self, x, targets=None):
        # 输入 x 的维度为 [Batch, 6, H, W]
        # 1. 拆分通道：前 3 通道为可见光，后 3 通道为红外
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        # 2. 并行双流特征提取
        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        # 3. 逐层自适应融合
        fused_feats = []
        for i, (f_rgb, f_ir) in enumerate(zip(feats_rgb, feats_ir)):
            # 融合 RGB 和 IR 特征，并应用光照引导
            f_fused = self.fusion_layers[i](f_rgb, f_ir)
            fused_feats.append(f_fused)

        # 4. 传入后续的 HybridEncoder 和 Decoder
        # 如果你正在切换到 O2-DFINE，此处的 Decoder 会处理旋转框分布回归 (ADR)
        x = self.encoder(fused_feats)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self