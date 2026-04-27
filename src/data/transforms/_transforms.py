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

# ==================== 1. 基础非几何变换 ====================
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
Normalize = register()(T.Normalize)

@register()
class EmptyTransform(T.Transform):
    def forward(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]

def _get_img_size(inputs):
    """从输入序列中动态寻找图像尺寸"""
    for x in inputs:
        if isinstance(x, (PIL.Image.Image, torch.Tensor, Image)):
            return (x.size if isinstance(x, PIL.Image.Image) else (x.shape[-1], x.shape[-2]))
    return None

# ==================== 2. 核心修正：结构感知几何变换 ====================

@register()
class Resize(T.Resize):
    def forward(self, *inputs: Any) -> Any:
        img_sz = _get_img_size(inputs)
        if img_sz is None: return inputs if len(inputs) > 1 else inputs[0]
        ow, oh = img_sz
        nh, nw = self.size if isinstance(self.size, (list, tuple)) else (self.size, self.size)
        
        outputs = []
        for x in inputs:
            if isinstance(x, (PIL.Image.Image, torch.Tensor, Image)):
                outputs.append(F.resize(x, self.size, interpolation=self.interpolation, antialias=self.antialias))
            elif isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5:
                target = x.copy()
                boxes = target["boxes"].clone()
                boxes[:, 0] *= (nw / ow); boxes[:, 1] *= (nh / oh); boxes[:, 2] *= (nw / ow); boxes[:, 3] *= (nh / oh)
                target["boxes"] = convert_to_tv_tensor(boxes, key="boxes", box_format=x["boxes"].format.value, spatial_size=(nh, nw))
                target["orig_size"] = torch.as_tensor([int(nw), int(nh)])
                outputs.append(target)
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        img_sz = _get_img_size(inputs)
        if img_sz is None: return inputs if len(inputs) > 1 else inputs[0]
        w, _ = img_sz

        outputs = []
        for x in inputs:
            if isinstance(x, (PIL.Image.Image, torch.Tensor, Image)):
                outputs.append(F.horizontal_flip(x))
            elif isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5:
                target = x.copy()
                boxes = target["boxes"].clone()
                boxes[:, 0] = w - boxes[:, 0]
                boxes[:, 4] = -boxes[:, 4]
                target["boxes"] = convert_to_tv_tensor(boxes, key="boxes", box_format=x["boxes"].format.value, spatial_size=x["boxes"].canvas_size)
                outputs.append(target)
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class RandomCrop(T.RandomCrop):
    def forward(self, *inputs: Any) -> Any:
        img_node = next((x for x in inputs if isinstance(x, (PIL.Image.Image, torch.Tensor, Image))), None)
        if img_node is None: return inputs if len(inputs) > 1 else inputs[0]
        
        top, left, height, width = self._get_params([img_node], self.size)
        outputs = []
        for x in inputs:
            if isinstance(x, (PIL.Image.Image, torch.Tensor, Image)):
                outputs.append(F.crop(x, top, left, height, width))
            elif isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5:
                target = x.copy()
                boxes = target["boxes"].clone()
                boxes[:, 0] -= left; boxes[:, 1] -= top
                keep = (boxes[:, 0] >= 0) & (boxes[:, 0] < width) & (boxes[:, 1] >= 0) & (boxes[:, 1] < height)
                target["boxes"] = convert_to_tv_tensor(boxes[keep], key="boxes", box_format=x["boxes"].format.value, spatial_size=(height, width))
                if "labels" in target: target["labels"] = target["labels"][keep]
                outputs.append(target)
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

# ==================== 3. 保护机制与必需类 ====================

@register()
class RandomZoomOut(T.RandomZoomOut):
    def forward(self, *inputs: Any) -> Any:
        # 检测到 OBB 则跳过防止越界报错
        if any(isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5 for x in inputs):
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)

@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def forward(self, *inputs: Any) -> Any:
        if any(isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5 for x in inputs):
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)

@register()
class SanitizeBoundingBoxes(SanitizeBoundingBoxes):
    def forward(self, *inputs: Any) -> Any:
        outputs = []
        for x in inputs:
            if isinstance(x, dict) and "boxes" in x and x["boxes"].shape[-1] == 5:
                target = x.copy()
                keep = (target["boxes"][:, 2] > 0) & (target["boxes"][:, 3] > 0)
                target["boxes"] = target["boxes"][keep]
                if "labels" in target: target["labels"] = target["labels"][keep]
                outputs.append(target)
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class ConvertPILImage(T.Transform):
    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype, self.scale = dtype, scale
    def forward(self, *inputs: Any) -> Any:
        outputs = []
        for x in inputs:
            if isinstance(x, PIL.Image.Image):
                res = F.pil_to_tensor(x)
                if self.dtype == "float32": res = res.float()
                if self.scale: res = res / 255.0
                outputs.append(Image(res))
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class PadToSize(T.Pad):
    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int): size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)
    def forward(self, *inputs: Any) -> Any:
        img = next((x for x in inputs if isinstance(x, (PIL.Image.Image, torch.Tensor, Image))), None)
        if img is None: return inputs if len(inputs) > 1 else inputs[0]
        sp = F.get_spatial_size(img)
        padding = [0, 0, self.size[0] - sp[1], self.size[1] - sp[0]]
        outputs = []
        for x in inputs:
            if isinstance(x, (PIL.Image.Image, torch.Tensor, Image)):
                outputs.append(F.pad(x, padding=padding, fill=self._fill[type(x)], padding_mode=self.padding_mode))
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class ConvertBoxes(T.Transform):
    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt, self.normalize = fmt, normalize
    def forward(self, *inputs: Any) -> Any:
        outputs = []
        for x in inputs:
            if isinstance(x, dict) and "boxes" in x:
                target = x.copy()
                boxes = target["boxes"]
                sz = getattr(boxes, _boxes_keys[1])
                if boxes.shape[-1] == 5:
                    c, a = boxes[:, :4], boxes[:, 4:]
                    if self.fmt: c = torchvision.ops.box_convert(c, boxes.format.value.lower(), self.fmt.lower())
                    if self.normalize: c = c / torch.tensor(sz[::-1]).tile(2).to(c.device)[None]
                    target["boxes"] = convert_to_tv_tensor(torch.cat([c, a], -1), "boxes", self.fmt.upper() or boxes.format.value, sz)
                else:
                    if self.fmt: boxes = torchvision.ops.box_convert(boxes, boxes.format.value.lower(), self.fmt.lower())
                    if self.normalize: boxes = boxes / torch.tensor(sz[::-1]).tile(2).to(boxes.device)[None]
                    target["boxes"] = convert_to_tv_tensor(boxes, "boxes", self.fmt.upper(), sz)
                outputs.append(target)
            else:
                outputs.append(x)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]