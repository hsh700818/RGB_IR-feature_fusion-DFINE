import PIL
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    
    BOX_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    LABEL_TEXT_COLOR = "white"
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i, (sample, target) in enumerate(zip(samples, targets)):
        # 1. 双流 6通道处理
        if sample.shape[0] == 6:
            sample_visualization = torch.cat([sample[:3, ...], sample[3:, ...]], dim=2)
            is_dual = True
            orig_w = sample.shape[-1]
        else:
            sample_visualization = sample
            is_dual = False
            orig_w = sample.shape[-1]
        
        orig_h = sample.shape[-2]

        # 2. 图像反归一化 (仅用于可视化颜色恢复)
        if normalized:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(sample.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(sample.device)
            sample_visualization = sample_visualization * std + mean

        sample_visualization = sample_visualization.clamp(0, 1)
        sample_visualization = to_pil_image(sample_visualization)
        draw = ImageDraw.Draw(sample_visualization)

        # 3. 绘制标注
        if 'boxes' in target:
            boxes = target['boxes'].clone()
            labels = target.get('labels', [0] * len(boxes))

            # 核心修复点：根据数值量级判断是否需要缩放
            # 如果最大值 <= 1.0，说明是归一化坐标，需要乘以图片尺寸
            is_normalized_boxes = boxes.max() <= 1.01 
            
            if box_fmt != 'xyxy':
                boxes = box_convert(boxes, in_fmt=box_fmt, out_fmt='xyxy')
            
            if is_normalized_boxes:
                boxes[:, [0, 2]] *= orig_w
                boxes[:, [1, 3]] *= orig_h

            # 最终检查：确保 x1 <= x2 且 y1 <= y2，防止 PIL 报错
            for box, label in zip(boxes, labels):
                x0, y0, x1, y1 = box.tolist()
                
                # 再次确保坐标顺序正确
                left = min(x0, x1)
                top = min(y0, y1)
                right = max(x0, x1)
                bottom = max(y0, y1)
                
                # 裁剪到图像边界
                left = max(0, min(left, orig_w - 1))
                right = max(0, min(right, orig_w - 1))
                top = max(0, min(top, orig_h - 1))
                bottom = max(0, min(bottom, orig_h - 1))

                # 如果右边界仍小于等于左边界，跳过绘制
                if right <= left or bottom <= top:
                    continue

                box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]
                draw.rectangle([left, top, right, bottom], outline=box_color, width=3)
                
                if is_dual:
                    draw.rectangle([left + orig_w, top, right + orig_w, bottom], outline=box_color, width=3)

        # 保存结果
        img_id = target.get('image_id', [i]).item() if isinstance(target.get('image_id'), torch.Tensor) else i
        save_path = Path(output_dir) / f"{split}_samples" / f"{img_id}_{i}.webp"
        sample_visualization.save(save_path)

def show_sample(sample):
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    image, target = sample
    if image.shape[0] == 6: image = image[:3, ...]
    if isinstance(image, PIL.Image.Image): image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)
    plt.imshow(annotated_image.permute(1, 2, 0))
    plt.show()