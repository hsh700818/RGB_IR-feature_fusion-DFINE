from pathlib import Path
from typing import Dict, List

import os
import math

import torch
import torch.utils.data
import torchvision
from PIL import ImageDraw, ImageFont
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_pil_image


torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _denormalize_3ch(image: torch.Tensor) -> torch.Tensor:
    """Denormalize one 3-channel RGB-like tensor for visualization only."""
    mean = torch.tensor(_IMAGENET_MEAN, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return image * std + mean


def _prepare_visual_image(sample: torch.Tensor, normalized: bool):
    """Prepare RGB/IR debug image and keep the single-modality width for box offsets."""
    if sample.shape[0] == 6:
        rgb = sample[:3, ...]
        ir = sample[3:, ...]
        if normalized:
            rgb = _denormalize_3ch(rgb)
            ir = _denormalize_3ch(ir)
        sample_visualization = torch.cat([rgb.clamp(0, 1), ir.clamp(0, 1)], dim=2)
        return sample_visualization, True, sample.shape[-1], sample.shape[-2]

    sample_visualization = sample
    if normalized and sample_visualization.shape[0] == 3:
        sample_visualization = _denormalize_3ch(sample_visualization)
    return sample_visualization.clamp(0, 1), False, sample.shape[-1], sample.shape[-2]


def get_rotated_vertices(cx, cy, w, h, angle):
    """
    Convert a rotated box in (cx, cy, w, h, angle) to four vertices.
    angle is expected in radians.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx = [-w / 2, w / 2, w / 2, -w / 2]
    dy = [-h / 2, -h / 2, h / 2, h / 2]

    vertices = []
    for x, y in zip(dx, dy):
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        vertices.append((rx, ry))
    return vertices


def _draw_boxes(draw, boxes, labels, orig_w, orig_h, box_fmt, x_offset=0):
    box_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    if boxes is None or len(boxes) == 0:
        return

    boxes = boxes.clone()
    labels = labels if labels is not None else [0] * len(boxes)
    is_normalized = boxes[:, :4].max() <= 1.01

    for box, label in zip(boxes, labels):
        box_color = box_colors[int(label) % len(box_colors)]

        if box.shape[-1] == 5:
            cx, cy, w, h, angle = box.tolist()
            if is_normalized:
                cx, w = cx * orig_w, w * orig_w
                cy, h = cy * orig_h, h * orig_h

            vertices = get_rotated_vertices(cx, cy, w, h, angle)
            if x_offset != 0:
                vertices = [(x + x_offset, y) for x, y in vertices]
            draw.polygon(vertices, outline=box_color, width=3)
        else:
            if box_fmt != "xyxy":
                box = box_convert(box.unsqueeze(0), in_fmt=box_fmt, out_fmt="xyxy").squeeze(0)
            x0, y0, x1, y1 = box.tolist()
            if is_normalized:
                x0, x1 = x0 * orig_w, x1 * orig_w
                y0, y1 = y0 * orig_h, y1 * orig_h

            draw.rectangle([x0 + x_offset, y0, x1 + x_offset, y1], outline=box_color, width=3)


def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)

    box_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    try:
        ImageFont.truetype("arial.ttf", 15)
    except Exception:
        ImageFont.load_default()

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization, is_dual, orig_w, orig_h = _prepare_visual_image(sample, normalized)
        img_pil = to_pil_image(sample_visualization)
        draw = ImageDraw.Draw(img_pil)

        if is_dual:
            rgb_boxes = target.get("rgb_boxes", target.get("boxes"))
            rgb_labels = target.get("rgb_labels", target.get("labels"))
            ir_boxes = target.get("ir_boxes", target.get("boxes"))
            ir_labels = target.get("ir_labels", target.get("labels"))
            _draw_boxes(draw, rgb_boxes, rgb_labels, orig_w, orig_h, box_fmt, x_offset=0)
            _draw_boxes(draw, ir_boxes, ir_labels, orig_w, orig_h, box_fmt, x_offset=orig_w)
        else:
            _draw_boxes(draw, target.get("boxes"), target.get("labels"), orig_w, orig_h, box_fmt, x_offset=0)

        image_id = target.get("image_id", [i])
        img_id = image_id.item() if isinstance(image_id, torch.Tensor) else i
        save_path = Path(output_dir) / f"{split}_samples" / f"sample_{img_id}_{i}.webp"
        img_pil.save(save_path)


def show_sample(sample, normalized=True):
    """
    Show one debug sample.
    RGB-IR 6-channel input is rendered as RGB on the left and IR on the right.
    """
    import matplotlib.pyplot as plt

    image, target = sample
    visualization_img, is_dual, orig_w, orig_h = _prepare_visual_image(image, normalized)

    img_pil = to_pil_image(visualization_img)
    draw = ImageDraw.Draw(img_pil)
    box_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]

    if "boxes" in target and len(target["boxes"]) > 0:
        if is_dual:
            _draw_boxes(draw, target.get("rgb_boxes", target.get("boxes")), target.get("rgb_labels", target.get("labels")), orig_w, orig_h, "cxcywh", x_offset=0)
            _draw_boxes(draw, target.get("ir_boxes", target.get("boxes")), target.get("ir_labels", target.get("labels")), orig_w, orig_h, "cxcywh", x_offset=orig_w)
        else:
            _draw_boxes(draw, target.get("boxes"), target.get("labels"), orig_w, orig_h, "cxcywh", x_offset=0)

    plt.figure(figsize=(16, 8))
    plt.imshow(img_pil)
    plt.title(f"Sample Debug View | Dual Stream: {is_dual}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
