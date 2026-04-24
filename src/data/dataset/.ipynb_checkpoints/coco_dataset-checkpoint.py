import os
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import faster_coco_eval.core.mask as coco_mask
from faster_coco_eval.utils.pytorch import FasterCocoDetection

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection"]

@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            # 注意：这里的 transforms 必须支持 Tensor 输入
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        # 1. 获取 RGB 图像和标注 (由基类返回 PIL Image 和 list)
        image_rgb, target_ann = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        
        # 2. 路径逻辑
        img_info = self.coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_folder, file_name)
        
        # 3. 加载对应的红外图像
        path_ir = image_path.replace('images_rgb', 'images_thermal')
        if not os.path.exists(path_ir):
            image_ir = image_rgb.copy() # 找不到则用 RGB 占位
        else:
            image_ir = Image.open(path_ir).convert('RGB')
        
        # 4. 预处理标注 (使用 RGB 图像的尺寸进行计算)
        target_dict = {"image_id": image_id, "image_path": image_path, "annotations": target_ann}
        if self.remap_mscoco_category:
            _, target_dict = self.prepare(image_rgb, target_dict, category2label=mscoco_category2label)
        else:
            _, target_dict = self.prepare(image_rgb, target_dict)
            
        # 5. 核心修改：转换为 Tensor [3, H, W] 后直接拼接为 [6, H, W]
        # 避开 PIL 无法处理 6 通道的限制
        t_rgb = F.to_tensor(image_rgb)
        t_ir = F.to_tensor(image_ir)
        img_6ch = torch.cat([t_rgb, t_ir], dim=0) 
        
        target_dict["idx"] = torch.tensor([idx])
        if "boxes" in target_dict:
            target_dict["boxes"] = convert_to_tv_tensor(
                target_dict["boxes"], key="boxes", spatial_size=image_rgb.size[::-1]
            )

        return img_6ch, target_dict

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s

# ==================== 辅助类与字典 ====================

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, **kwargs):
        # 确保能处理 PIL 或 Tensor 的尺寸获取
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]

        image_id = torch.tensor([target["image_id"]])
        image_path = target["image_path"]
        anno = [obj for obj in target["annotations"] if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get("category2label", None)
        labels = torch.tensor([category2label[obj["category_id"]] if category2label else obj["category_id"] for obj in anno], dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        target = {
            "boxes": boxes[keep], 
            "labels": labels[keep], 
            "image_id": image_id, 
            "image_path": image_path,
            "area": torch.tensor([obj["area"] for obj in anno])[keep],
            "iscrowd": torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep],
            "orig_size": torch.as_tensor([int(w), int(h)])
        }
        return image, target

mscoco_category2name = {1: "fire", 2: "smoke"}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}