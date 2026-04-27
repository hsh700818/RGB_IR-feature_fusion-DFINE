import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from typing import Any, Dict, List, Optional
import PIL.Image

from ...core import register
from .._misc import (
    BoundingBoxes, Image, Mask, SanitizeBoundingBoxes, 
    Video, _boxes_keys, convert_to_tv_tensor
)

torchvision.disable_beta_transforms_warning()

# ==================== 1. 注册标准 torchvision 变换 ====================

RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)

# ==================== 2. 支持 OBB (5维) 的几何变换 ====================

@register()
class Resize(T.Resize):
    """适配旋转框 (5维) 的 Resize：缩放前4位坐标，保持第5位角度不变"""
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes) and inpt.shape[-1] == 5:
            oh, ow = inpt.canvas_size
            nh, nw = self.size if isinstance(self.size, (list, tuple)) else (self.size, self.size)
            
            # 手动计算缩放比例并克隆数据
            res = inpt.clone()
            # 缩放 cx, cy, w, h
            res[:, 0] *= (nw / ow)
            res[:, 1] *= (nh / oh)
            res[:, 2] *= (nw / ow)
            res[:, 3] *= (nh / oh)
            # 第5位 angle 不随图像尺寸缩放而改变
            
            return convert_to_tv_tensor(
                res, key="boxes", 
                box_format=inpt.format.value, 
                spatial_size=(nh, nw)
            )
        # 非5维数据走原生逻辑
        return super()._transform(inpt, params)

@register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """适配旋转框 (5维) 的水平翻转：中心点反转 + 角度取负号"""
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes) and inpt.shape[-1] == 5:
            res = inpt.clone()
            # 翻转中心点 cx (假设坐标已归一化，D-FINE 默认在 flip 前已归一化)
            res[:, 0] = 1.0 - res[:, 0] 
            # 翻转角度
            res[:, 4] = -res[:, 4]

            return convert_to_tv_tensor(
                res, key="boxes", 
                box_format=inpt.format.value, 
                spatial_size=getattr(inpt, _boxes_keys[1])
            )
        return super()._transform(inpt, params)

# ==================== 3. 必需的辅助变换类  ====================

@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)

@register()
class EmptyTransform(T.Transform):
    def forward(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]

@register()
class ConvertPILImage(T.Transform):
    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype, self.scale = dtype, scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, PIL.Image.Image):
            inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.0
        return Image(inpt)

@register()
class PadToSize(T.Pad):
    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int): size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.pad(inpt, padding=params["padding"], fill=self._fill[type(inpt)], padding_mode=self.padding_mode)

@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)
    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt, self.normalize = fmt, normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if inpt.shape[-1] == 5:
            coords, angle = inpt[:, :4], inpt[:, 4:]
            if self.fmt:
                in_fmt = inpt.format.value.lower()
                coords = torchvision.ops.box_convert(coords, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            if self.normalize:
                coords = coords / torch.tensor(spatial_size[::-1]).tile(2).to(coords.device)[None]
            res = torch.cat([coords, angle], dim=-1)
            return convert_to_tv_tensor(res, key="boxes", box_format=self.fmt.upper() if self.fmt else inpt.format.value, spatial_size=spatial_size)
        
        if self.fmt:
            inpt = torchvision.ops.box_convert(inpt, in_fmt=inpt.format.value.lower(), out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size)
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2).to(inpt.device)[None]
        return inpt