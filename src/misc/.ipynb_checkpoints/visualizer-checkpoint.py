import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.transforms.functional import to_pil_image
from torchvision.ops import box_convert
from pathlib import Path
import os
import math
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def get_rotated_vertices(cx, cy, w, h, angle):
    """
    将旋转框 5 参数转换为矩形的 4 个顶点坐标。
    angle: 弧度制
    """
    import math
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 矩形未旋转时相对于中心点的四个顶点
    dx = [-w/2,  w/2, w/2, -w/2]
    dy = [-h/2, -h/2, h/2,  h/2]
    
    vertices = []
    for x, y in zip(dx, dy):
        # 应用旋转矩阵并平移回中心点
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        vertices.append((rx, ry))
    return vertices

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    
    BOX_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i, (sample, target) in enumerate(zip(samples, targets)):
        # 1. 图像处理 (支持双流拼接显示)
        if sample.shape[0] == 6:
            sample_visualization = torch.cat([sample[:3, ...], sample[3:, ...]], dim=2)
            is_dual = True
            orig_w = sample.shape[-1]
        else:
            sample_visualization = sample
            is_dual = False
            orig_w = sample.shape[-1]
        
        orig_h = sample.shape[-2]

        if normalized:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(sample.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(sample.device)
            sample_visualization = sample_visualization * std + mean

        sample_visualization = sample_visualization.clamp(0, 1)
        img_pil = to_pil_image(sample_visualization)
        draw = ImageDraw.Draw(img_pil)

        # 2. 绘制 OBB 标注
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            labels = target.get('labels', [0] * len(boxes))

            # 归一化坐标还原到像素坐标
            is_normalized = boxes[:, :4].max() <= 1.01
            
            for idx, (box, label) in enumerate(zip(boxes, labels)):
                box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]
                
                # --- 核心逻辑：区分 OBB 和 HBB ---
                if box.shape[-1] == 5:
                    # 旋转框格式 [cx, cy, w, h, angle]
                    cx, cy, w, h, angle = box.tolist()
                    if is_normalized:
                        cx, w = cx * orig_w, w * orig_w
                        cy, h = cy * orig_h, h * orig_h
                    
                    # 获取四个顶点
                    vertices = get_rotated_vertices(cx, cy, w, h, angle)
                    
                    # 画左侧图的旋转框
                    draw.polygon(vertices, outline=box_color, width=3)
                    
                    # 如果是双流，右侧图也画上
                    if is_dual:
                        v_dual = [(x + orig_w, y) for x, y in vertices]
                        draw.polygon(v_dual, outline=box_color, width=3)
                else:
                    # 普通水平框处理逻辑保持不变
                    if box_fmt != 'xyxy':
                        box = box_convert(box.unsqueeze(0), in_fmt=box_fmt, out_fmt='xyxy').squeeze(0)
                    x0, y0, x1, y1 = box.tolist()
                    if is_normalized:
                        x0, x1 = x0 * orig_w, x1 * orig_w
                        y0, y1 = y0 * orig_h, y1 * orig_h
                    
                    draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)
                    if is_dual:
                        draw.rectangle([x0 + orig_w, y0, x1 + orig_w, y1], outline=box_color, width=3)

        # 3. 保存
        img_id = target.get('image_id', [i]).item() if isinstance(target.get('image_id'), torch.Tensor) else i
        save_path = Path(output_dir) / f"{split}_samples" / f"sample_{img_id}_{i}.webp"
        img_pil.save(save_path)

def get_rotated_vertices(cx, cy, w, h, angle):
    """
    将旋转框 5 参数转换为矩形的 4 个顶点坐标。
    angle: 弧度制
    """
    import math
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 矩形未旋转时相对于中心点的四个顶点
    dx = [-w/2,  w/2, w/2, -w/2]
    dy = [-h/2, -h/2, h/2,  h/2]
    
    vertices = []
    for x, y in zip(dx, dy):
        # 应用旋转矩阵并平移回中心点
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        vertices.append((rx, ry))
    return vertices

def show_sample(sample):
    """
    显示单个训练样本。
    1. 自动识别并拼接 6 通道 RGB-IR 图像。
    2. 支持 5 参数旋转框 (OBB) 的倾斜绘制 (Polygon)。
    3. 支持归一化坐标自动还原。
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    from PIL import ImageDraw
    import torch

    image, target = sample
    
    # --- 1. 图像预处理 (处理双流 6通道) ---
    # 如果图像是 (6, H, W)，则将可见光和红外图像在宽度上拼接显示
    if image.shape[0] == 6:
        visualization_img = torch.cat([image[:3, ...], image[3:, ...]], dim=2)
        is_dual = True
        orig_w = image.shape[-1] # 单张图的宽度
    else:
        visualization_img = image
        is_dual = False
        orig_w = image.shape[-1]
    
    orig_h = image.shape[-2]

    # --- 2. 反归一化 (用于颜色恢复) ---
    # 使用 ImageNet 的标准均值/方差，如果你的 YML 里有自定义请对应修改
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    visualization_img = visualization_img * std + mean
    visualization_img = visualization_img.clamp(0, 1)

    # 转换为 PIL 用于绘图
    img_pil = to_pil_image(visualization_img)
    draw = ImageDraw.Draw(img_pil)
    BOX_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    # --- 3. 绘制标注框 ---
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes'].clone()
        labels = target.get('labels', [0] * len(boxes))
        
        # 判断坐标是否为 0~1 的归一化格式
        is_normalized = boxes[:, :4].max() <= 1.01

        for box, label in zip(boxes, labels):
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]
            
            # 情况 A: 旋转框 (5 参数: cx, cy, w, h, angle)
            if box.shape[-1] == 5:
                cx, cy, w, h, angle = box.tolist()
                
                if is_normalized:
                    cx, w = cx * orig_w, w * orig_w
                    cy, h = cy * orig_h, h * orig_h
                
                # 计算旋转顶点
                vertices = get_rotated_vertices(cx, cy, w, h, angle)
                
                # 绘制旋转多边形
                draw.polygon(vertices, outline=box_color, width=3)
                # 如果是双流，在右侧红外图对应的位置也绘制
                if is_dual:
                    v_dual = [(x + orig_w, y) for x, y in vertices]
                    draw.polygon(v_dual, outline=box_color, width=3)
            
            # 情况 B: 水平框 (4 参数: cx, cy, w, h)
            else:
                from torchvision.ops import box_convert
                box_xyxy = box_convert(box.unsqueeze(0), in_fmt='cxcywh', out_fmt='xyxy').squeeze(0)
                x0, y0, x1, y1 = box_xyxy.tolist()
                
                if is_normalized:
                    x0, x1 = x0 * orig_w, x1 * orig_w
                    y0, y1 = y0 * orig_h, y1 * orig_h
                
                draw.rectangle([x0, y0, x1, y1], outline=box_color, width=3)
                if is_dual:
                    draw.rectangle([x0 + orig_w, y0, x1 + orig_w, y1], outline=box_color, width=3)

    # --- 4. 使用 Matplotlib 显示 ---
    plt.figure(figsize=(16, 8))
    plt.imshow(img_pil)
    plt.title(f"Sample Debug View | Dual Stream: {is_dual}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()