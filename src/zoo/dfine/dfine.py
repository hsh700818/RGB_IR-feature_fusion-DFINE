import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None

__all__ = ["DFINE"]


class GLSA(nn.Module):
    """
    Global-local spatial attention for small aerial targets.
    It keeps the original residual path and uses lightweight spatial/channel
    weights to refine fused RGB-IR features.
    """

    def __init__(self, ch):
        super().__init__()
        hidden_ch = max(ch // 8, 16)

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), groups=ch)
        self.conv_w = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        x_h = self.conv_h(self.avg_pool_h(x))
        x_w = self.conv_w(self.avg_pool_w(x))
        x_c = self.channel_att(x)
        return identity * x_h.sigmoid() * x_w.sigmoid() * x_c


class AlignmentAwareFusion(nn.Module):
    """
    Residual deformable alignment for RGB-IR feature misalignment.

    The learned offset is bounded and initialized to zero. The deformable branch
    is attached through a zero-initialized residual scale, so replacing the old
    fusion block does not immediately disturb a pretrained or partially trained
    model.
    """

    def __init__(self, ch, max_offset=4.0):
        super().__init__()
        self.max_offset = float(max_offset)

        self.offset_conv = nn.Conv2d(ch * 2, 2 * 3 * 3, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(ch * 2, 1 * 3 * 3, kernel_size=3, padding=1)

        if DeformConv2d is not None:
            self.dcn = DeformConv2d(ch, ch, kernel_size=3, padding=1, bias=False)
            self.fallback_align = None
        else:
            # Fallback keeps the code runnable when torchvision was built
            # without deformable convolution support.
            self.dcn = None
            self.fallback_align = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            )

        self.align_scale = nn.Parameter(torch.zeros(1))
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, ch, 1),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.mask_conv.weight, 0.0)
        nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, f_rgb, f_ir):
        feat_cat = torch.cat([f_rgb, f_ir], dim=1)

        if self.dcn is not None:
            offsets = torch.tanh(self.offset_conv(feat_cat)) * self.max_offset
            mask = torch.sigmoid(self.mask_conv(feat_cat))
            aligned_delta = self.dcn(f_ir, offsets, mask)
        else:
            aligned_delta = self.fallback_align(f_ir)

        f_ir_aligned = f_ir + self.align_scale * aligned_delta

        # A light modality gate avoids forcing IR-aligned features to dominate
        # all scenes. In bright scenes it can keep more RGB detail; in low-light
        # scenes it can lean toward IR.
        gate = self.fusion_gate(torch.cat([f_rgb, f_ir_aligned], dim=1))
        return f_rgb * gate + f_ir_aligned * (1.0 - gate)


class CrossModalInteraction(nn.Module):
    """
    Memory-safe cross-modal interaction.

    The Gemini-style full HW x HW attention is expensive on P3. This variant
    performs the cross-modal attention on adaptively pooled tokens and then
    upsamples the complementary context back to the original feature size.
    """

    def __init__(self, ch, reduction=4, max_tokens=400):
        super().__init__()
        attn_ch = max(ch // reduction, 32)
        self.max_tokens = int(max_tokens)

        self.query_conv = nn.Conv2d(ch, attn_ch, 1)
        self.key_conv = nn.Conv2d(ch, attn_ch, 1)
        self.value_conv = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pool_if_needed(self, x):
        _, _, h, w = x.shape
        if h * w <= self.max_tokens:
            return x, (h, w)

        pooled_hw = max(int(self.max_tokens ** 0.5), 1)
        return F.adaptive_avg_pool2d(x, (pooled_hw, pooled_hw)), (pooled_hw, pooled_hw)

    def forward(self, f_main, f_aux):
        b, c, h, w = f_main.shape

        f_main_pool, _ = self._pool_if_needed(f_main)
        f_aux_pool, _ = self._pool_if_needed(f_aux)

        _, _, hp, wp = f_main_pool.shape
        token_num = hp * wp

        proj_query = self.query_conv(f_main_pool).flatten(2).transpose(1, 2)
        proj_key = self.key_conv(f_aux_pool).flatten(2)
        energy = torch.bmm(proj_query, proj_key) / (proj_key.shape[1] ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value_conv(f_aux_pool).flatten(2)
        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out = out.view(b, c, hp, wp)

        if token_num != h * w:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return f_main + self.gamma * out


class AdvancedMultimodalFusion(nn.Module):
    """
    AAF + memory-safe CMI + illumination-aware fusion.

    It replaces the previous IlluminationAwareFusion while keeping compatible
    names for illum_extract, fusion_conv and glsa so older tuning checkpoints can
    still reuse part of their fusion weights.
    """

    def __init__(self, ch):
        super().__init__()
        self.alignment = AlignmentAwareFusion(ch)
        self.rgb_enhance = CrossModalInteraction(ch)
        self.ir_enhance = CrossModalInteraction(ch)

        self.illum_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // 4, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // 4, 16), 1, 1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Conv2d(ch * 2, ch, 1)
        self.glsa = GLSA(ch)

    def forward(self, f_rgb, f_ir):
        f_ir_aligned = self.alignment(f_rgb, f_ir)

        f_rgb_ext = self.rgb_enhance(f_rgb, f_ir_aligned)
        f_ir_ext = self.ir_enhance(f_ir_aligned, f_rgb)

        illum_score = self.illum_extract(f_rgb)
        f_cat = torch.cat(
            [f_rgb_ext * illum_score, f_ir_ext * (2.0 - illum_score)],
            dim=1,
        )
        f_fused = self.fusion_conv(f_cat)

        return self.glsa(f_fused)


# Backward-compatible alias for configs or checkpoints that refer to the old name.
IlluminationAwareFusion = AdvancedMultimodalFusion


@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = copy.deepcopy(backbone)
        self.encoder = encoder
        self.decoder = decoder

        in_channels = encoder.in_channels
        self.fusion_layers = nn.ModuleList(
            [AdvancedMultimodalFusion(ch) for ch in in_channels]
        )

    def forward(self, x, targets=None):
        # Input shape: [B, 6, H, W]. The first 3 channels are RGB and the
        # last 3 channels are IR.
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        fused_feats = []
        for f_rgb, f_ir, fusion in zip(feats_rgb, feats_ir, self.fusion_layers):
            fused_feats.append(fusion(f_rgb, f_ir))

        x = self.encoder(fused_feats)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
