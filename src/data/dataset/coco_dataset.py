import os
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset
from faster_coco_eval.utils.pytorch import FasterCocoDetection

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection"]

@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        # 显式初始化基类
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        
        # 标注预处理器
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        """
        返回拼接后的 6 通道图像 Tensor [6, H, W] 和处理后的标注
        """
        img, target = self.load_item(idx)
        if self._transforms is not None:
            # 这里的 transforms 必须适配 Tensor 输入 (见 dataloader.yml 修改建议)
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        # 1. 获取 ID 和文件名 (从基类获取可见光图像信息)
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        
        # 2. 读取可见光图像 (RGB)
        # 路径例如: /root/autodl-tmp/DroneVehicle/train/trainimg/00001.jpg
        path_v = os.path.join(self.img_folder, file_name)
        img_v = Image.open(path_v).convert('RGB')
        
        # 3. 自动定位对应的红外图像 (IR)
        # 替换逻辑：针对你的命名，将文件夹名中的 'img' 替换为 'imgr'
        # 这样可以同时处理 trainimg -> trainimgr 和 valimg -> valimgr
        path_ir = path_v.replace('img', 'imgr')
        
        if not os.path.exists(path_ir):
            # 如果文件夹映射失败，尝试文件名替换（兼容性备份）
            path_ir = path_v.replace('.jpg', 'r.jpg')
            
        if not os.path.exists(path_ir):
            raise FileNotFoundError(f"未找到对应的红外图像，请检查路径映射: {path_ir}")
        
        img_ir = Image.open(path_ir).convert('RGB')

        # 4. 获取原始标注 (基类从 JSON 加载)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target_raw = self.coco.loadAnns(ann_ids)
        target_dict = {"image_id": image_id, "image_path": path_v, "annotations": target_raw}

        # 5. 处理标签和边界框格式 (映射类别)
        if self.remap_mscoco_category:
            _, target_dict = self.prepare(img_v, target_dict, category2label=mscoco_category2label)
        else:
            _, target_dict = self.prepare(img_v, target_dict)

        # 6. 核心：转换为 Tensor 并拼接为 6 通道 [6, H, W]
        # F.to_tensor 会自动将 [0, 255] 缩放到 [0.0, 1.0]
        t_v = F.to_tensor(img_v)
        t_ir = F.to_tensor(img_ir)
        img_6ch = torch.cat([t_v, t_ir], dim=0) 

        # 7. 格式化标注供 D-FINE 使用
        target_dict["idx"] = torch.tensor([idx])
        if "boxes" in target_dict:
            target_dict["boxes"] = convert_to_tv_tensor(
                target_dict["boxes"], key="boxes", spatial_size=img_v.size[::-1]
            )

        return img_6ch, target_dict

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s

# ==================== 辅助类 ====================

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, **kwargs):
        # image 可能是 PIL 或 Tensor
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            w, h = image.shape[-1], image.shape[-2]

        image_id = torch.tensor([target["image_id"]])
        image_path = target["image_path"]
        
        # 过滤无效标注
        anno = [obj for obj in target["annotations"] if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
        # COCO [x, y, w, h] -> [x1, y1, x2, y2]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 类别转换
        category2label = kwargs.get("category2label", None)
        labels = torch.tensor(
            [category2label[obj["category_id"]] if category2label else obj["category_id"] for obj in anno], 
            dtype=torch.int64
        )

        # 过滤非法框
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        
        target_res = {
            "boxes": boxes[keep], 
            "labels": labels[keep], 
            "image_id": image_id, 
            "image_path": image_path,
            "area": torch.tensor([obj["area"] for obj in anno])[keep],
            "iscrowd": torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep],
            "orig_size": torch.as_tensor([int(w), int(h)])
        }
        return image, target_res

# ==================== DroneVehicle 类别字典 ====================
# 基于你之前 Counter 统计的 5 个标准类
mscoco_category2name = {
    1: "car", 
    2: "truck", 
    3: "bus", 
    4: "van", 
    5: "freight car"
}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}