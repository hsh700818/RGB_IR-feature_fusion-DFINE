import os
import json
import cv2
from tqdm import tqdm

def yolo_to_coco():
    # 1. 基础配置
    root_path = '/root/autodl-tmp/D-Fire/data'
    # class 0: 烟雾 (smoke), class 1: 火焰 (fire)
    classes = ['smoke', 'fire'] 
    subsets = ['train', 'val']

    for subset in subsets:
        print(f"正在处理 {subset} 数据集...")
        img_dir = os.path.join(root_path, subset, 'images')
        label_dir = os.path.join(root_path, subset, 'labels')
        
        # 检查路径是否存在
        if not os.path.exists(img_dir):
            print(f"跳过 {subset}: 找不到图片目录 {img_dir}")
            continue

        coco_output = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
        }

        ann_id = 0
        img_id = 0
        
        # 遍历图片
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_name in tqdm(image_files):
            # 获取图片尺寸
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width, _ = img.shape

            # 添加图片信息
            coco_output["images"].append({
                "file_name": img_name,
                "id": img_id,
                "width": width,
                "height": height
            })

            # 读取对应的 YOLO 标签
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(label_dir, txt_name)

            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        cls_id, cx, cy, nw, nh = map(float, parts)
                        
                        # YOLO (归一化中心坐标) -> COCO (绝对像素坐标 [x_min, y_min, w, h])
                        w = nw * width
                        h = nh * height
                        x_min = (cx * width) - (w / 2)
                        y_min = (cy * height) - (h / 2)

                        coco_output["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cls_id),
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        ann_id += 1
            
            img_id += 1

        # 将 JSON 保存到各自的子文件夹下
        output_file = os.path.join(root_path, subset, f'instances_{subset}.json')
        with open(output_file, 'w') as f:
            json.dump(coco_output, f)
        print(f"成功生成: {output_file}")

if __name__ == "__main__":
    yolo_to_coco()