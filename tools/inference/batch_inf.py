import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig

def batch_process(config_path, resume_path, input_dir, output_dir, device='cuda', thrh=0.4):
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载配置和模型
    cfg = YAMLConfig(config_path, resume=resume_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume_path, map_location='cpu')
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device).eval()
    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    # 3. 遍历文件夹
    img_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"检测到 {len(img_list)} 张图片，开始推理...")

    for img_name in tqdm(img_list):
        img_path = os.path.join(input_dir, img_name)
        im_pil = Image.open(img_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)

        # 绘制并保存
        draw = ImageDraw.Draw(im_pil)
        scr = scores[0]
        mask = scr > thrh
        curr_boxes = boxes[0][mask]
        curr_labels = labels[0][mask]
        curr_scores = scr[mask]

        for j, b in enumerate(curr_boxes):
            draw.rectangle(list(b), outline="red", width=3)
            draw.text((b[0], b[1]), text=f"{int(curr_labels[j])} {round(curr_scores[j].item(), 2)}", fill="blue")

        im_pil.save(os.path.join(output_dir, f"res_{img_name}"))

if __name__ == "__main__":
    batch_process(
        config_path='configs/dfine/custom/dfine_hgnetv2_n_dfire.yml',
        resume_path='./output/dfine_hgnetv2_n_dfire_160e/best_stg2.pth',
        input_dir='/root/autodl-tmp/D-Fire/data/test/images/',
        output_dir='./inference_results/', # 结果会存在这里
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )