import os
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageFile
import numpy as np

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset
from faster_coco_eval.utils.pytorch import FasterCocoDetection

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None
# Some DroneVehicle RGB/IR images may be partially truncated after upload or unzip.
# This prevents PIL from crashing on a small truncated tail such as "137 bytes not processed".
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ["CocoDetection", "mscoco_category2name", "mscoco_category2label", "mscoco_label2category"]

# DroneVehicle 类别定义
mscoco_category2name = {1: "car", 2: "truck", 3: "bus", 4: "van", 5: "freight car"}
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())} # 映射到 0-4
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self._filter_missing_rgb_ir_pairs()

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def _filter_missing_rgb_ir_pairs(self):
        """Drop COCO image ids that do not have both visible and infrared files."""
        kept_ids = []
        missing = []
        for image_id in list(self.ids):
            img_info = self.coco.loadImgs(image_id)[0]
            file_name = img_info['file_name']
            try:
                path_v = self._resolve_visible_path(file_name)
                self._resolve_ir_path(path_v)
                kept_ids.append(image_id)
            except FileNotFoundError as exc:
                missing.append((file_name, str(exc).split('\n')[0]))

        if missing:
            print(
                f"[CocoDetection] Skip {len(missing)} images without RGB-IR pairs "
                f"from {self.ann_file}. Keep {len(kept_ids)} images."
            )
            for file_name, reason in missing[:20]:
                print(f"[CocoDetection] missing pair: {file_name} | {reason}")
        self.ids = kept_ids

    def _resolve_visible_path(self, file_name):
        """Resolve visible image paths for common DroneVehicle COCO layouts."""
        base = os.path.basename(file_name)
        candidates = []

        if os.path.isabs(file_name):
            candidates.append(file_name)
        else:
            candidates.append(os.path.join(self.img_folder, file_name))

        candidates.extend([
            os.path.join(self.img_folder, base),
            os.path.join(self.img_folder, 'img', base),
            os.path.join(self.img_folder, 'trainimg', base),
            os.path.join(self.img_folder, 'valimg', base),
            os.path.join(self.img_folder, 'testimg', base),
            os.path.join(os.path.dirname(self.img_folder), 'img', base),
            os.path.join(os.path.dirname(self.img_folder), 'trainimg', base),
            os.path.join(os.path.dirname(self.img_folder), 'valimg', base),
            os.path.join(os.path.dirname(self.img_folder), 'testimg', base),
        ])

        stem, ext = os.path.splitext(base)
        for suffix in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
            alt = stem + suffix
            candidates.extend([
                os.path.join(self.img_folder, alt),
                os.path.join(self.img_folder, 'img', alt),
                os.path.join(self.img_folder, 'trainimg', alt),
                os.path.join(self.img_folder, 'valimg', alt),
                os.path.join(self.img_folder, 'testimg', alt),
                os.path.join(os.path.dirname(self.img_folder), 'img', alt),
                os.path.join(os.path.dirname(self.img_folder), 'trainimg', alt),
                os.path.join(os.path.dirname(self.img_folder), 'valimg', alt),
                os.path.join(os.path.dirname(self.img_folder), 'testimg', alt),
            ])

        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)

        for p in unique_candidates:
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            "未找到可见光图像，COCO file_name='{}'。已尝试路径：\n{}".format(
                file_name, "\n".join(unique_candidates[:30])
            )
        )

    @staticmethod
    def _replace_dir_token(path, src_token, dst_token):
        """
        Replace directory token safely without touching unrelated substrings.
        Example: /a/valimg/0001.jpg -> /a/valimgr/0001.jpg
        """
        norm = os.path.normpath(path)
        parts = norm.split(os.sep)
        replaced = [dst_token if p == src_token else p for p in parts]
        return os.sep.join(replaced)

    def _resolve_ir_path(self, path_v):
        """Resolve paired infrared image paths for common DroneVehicle layouts."""
        directory = os.path.dirname(path_v)
        base = os.path.basename(path_v)
        stem, ext = os.path.splitext(base)
        parent = os.path.dirname(directory)

        candidates = [
            self._replace_dir_token(path_v, 'trainimg', 'trainimgr'),
            self._replace_dir_token(path_v, 'valimg', 'valimgr'),
            self._replace_dir_token(path_v, 'testimg', 'testimgr'),
            path_v.replace('/img/', '/imgr/'),
            path_v.replace('\\img\\', '\\imgr\\'),
            os.path.join(self._replace_dir_token(directory, 'trainimg', 'trainimgr'), base),
            os.path.join(self._replace_dir_token(directory, 'valimg', 'valimgr'), base),
            os.path.join(self._replace_dir_token(directory, 'testimg', 'testimgr'), base),
            os.path.join(self._replace_dir_token(directory, 'img', 'imgr'), base),
            os.path.join(parent, 'imgr', base),
            os.path.join(parent, 'trainimgr', base),
            os.path.join(parent, 'valimgr', base),
            os.path.join(parent, 'testimgr', base),
            os.path.join(directory, stem + 'r' + ext),
            path_v.replace('.jpg', 'r.jpg'),
            path_v.replace('.JPG', 'r.JPG'),
            path_v.replace('.jpeg', 'r.jpeg'),
            path_v.replace('.JPEG', 'r.JPEG'),
            path_v.replace('.png', 'r.png'),
            path_v.replace('.PNG', 'r.PNG'),
            path_v.replace('.bmp', 'r.bmp'),
            path_v.replace('.BMP', 'r.BMP'),
        ]

        for suffix in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']:
            candidates.extend([
                os.path.join(parent, 'imgr', stem + suffix),
                os.path.join(parent, 'trainimgr', stem + suffix),
                os.path.join(parent, 'valimgr', stem + suffix),
                os.path.join(parent, 'testimgr', stem + suffix),
                os.path.join(parent, 'imgr', stem + 'r' + suffix),
                os.path.join(parent, 'trainimgr', stem + 'r' + suffix),
                os.path.join(parent, 'valimgr', stem + 'r' + suffix),
                os.path.join(parent, 'testimgr', stem + 'r' + suffix),
            ])

        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)

        src_abs = os.path.abspath(path_v)
        for p in unique_candidates:
            # Never allow IR to fall back to the same file as visible image.
            if os.path.abspath(p) == src_abs:
                continue
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            "未找到红外图像，可见光图像路径='{}'。已尝试路径：\n{}".format(
                path_v, "\n".join(unique_candidates[:30])
            )
        )

    def load_item(self, idx):
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        
        path_v = self._resolve_visible_path(file_name)
        try:
            img_v = Image.open(path_v).convert('RGB')
        except OSError as e:
            raise OSError(f"可见光图像读取失败，可能文件损坏: {path_v}") from e
        
        path_ir = self._resolve_ir_path(path_v)
            
        try:
            img_ir = Image.open(path_ir).convert('RGB')
        except OSError as e:
            raise OSError(f"红外图像读取失败，可能文件损坏: {path_ir}") from e

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        target_raw = self.coco.loadAnns(ann_ids)
        target_dict = {"image_id": image_id, "image_path": path_v, "annotations": target_raw}

        # 根据 yml 配置决定是否重映射标签
        if self.remap_mscoco_category:
            _, target_dict = self.prepare(img_v, target_dict, category2label=mscoco_category2label)
        else:
            _, target_dict = self.prepare(img_v, target_dict)

        img_6ch = torch.cat([F.to_tensor(img_v), F.to_tensor(img_ir)], dim=0) 
        target_dict["idx"] = torch.tensor([idx])
        
        if "boxes" in target_dict:
            # ConvertCocoPolysToMask returns rotated boxes in (cx, cy, w, h, angle).
            # Mark the first four coordinates as CXCYWH so ConvertBoxes(normalize=True)
            # divides (cx, w) by image width and (cy, h) by image height correctly.
            target_dict["boxes"] = convert_to_tv_tensor(
                target_dict["boxes"], key="boxes", box_format="cxcywh", spatial_size=img_v.size[::-1]
            )
        return img_6ch, target_dict

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, **kwargs):
        from ...zoo.dfine.box_ops import poly2rbox # 局部导入打破循环依赖
        
        w, h = (image.size if isinstance(image, Image.Image) else (image.shape[-1], image.shape[-2]))
        image_id = torch.tensor([target["image_id"]])
        anno = [obj for obj in target["annotations"] if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = []
        for obj in anno:
            if "segmentation" in obj and len(obj["segmentation"]) > 0 and len(obj["segmentation"][0]) == 8:
                poly = torch.tensor(obj["segmentation"][0], dtype=torch.float32)
                boxes.append(poly2rbox(poly)) 
            else:
                b = obj["bbox"]
                boxes.append(torch.tensor([b[0]+b[2]/2, b[1]+b[3]/2, b[2], b[3], 0.0]))

        if len(boxes) > 0:
            boxes = torch.stack(boxes).reshape(-1, 5)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)

        # 【核心修正】：严格映射标签 ID，防止 1-5 越界
        category2label = kwargs.get("category2label", mscoco_category2label)
        labels = []
        for obj in anno:
            cat_id = obj["category_id"]
            # 优先从映射表读，若无则 -1 容错
            labels.append(category2label.get(cat_id, cat_id - 1))
            
        labels = torch.tensor(labels, dtype=torch.int64)
        
        keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        
        target_res = {
            "boxes": boxes[keep], 
            "labels": labels[keep], 
            "image_id": image_id, 
            "image_path": target["image_path"],
            "area": torch.tensor([obj["area"] for obj in anno])[keep], 
            "iscrowd": torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep],
            "orig_size": torch.as_tensor([int(w), int(h)])
        }
        return image, target_res
