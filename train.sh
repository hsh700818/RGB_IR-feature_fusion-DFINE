#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# 缓解 PyTorch 显存碎片化导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 训练配置
CONFIG="configs/dfine/dfine_hgnetv2_s_dronevehicle.yml"
OUTPUT_DIR="./output/dronevehicle_s_obb_fusion_v3"
TUNING_PATH="weights_1/dfine_s_coco.pth"
# RESUME_PATH="./output/dronevehicle_s_obb_fusion_restart/checkpoint0011.pth"
UPDATE_PARA="train_dataloader.total_batch_size=18 val_dataloader.total_batch_size=18"

# 3. 启动前检查
if [ ! -f "$CONFIG" ]; then
    echo "配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$TUNING_PATH" ]; then
    echo "预训练权重不存在: $TUNING_PATH"
    echo "若不是重头训练，可忽略该提示。-t"
    exit 1
fi

# if [ ! -f "$RESUME_PATH" ]; then
#     echo "断点权重不存在: $RESUME_PATH"
#     echo "若不是断点续训，可忽略该提示。-r"
#     exit 1
# fi

mkdir -p "$OUTPUT_DIR"

echo "==================== 开始 D-FINE RGB-IR OBB 重新训练 ===================="
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "预训练权重: $TUNING_PATH"
echo "说明: 本脚本使用 -t 加载 COCO 预训练权重重新训练，不使用 -r resume。"

# 4. 执行训练
python train.py \
    -c "$CONFIG" \
    -t "$TUNING_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --use-amp \
    --seed 42 \
    # -r "$RESUME_PATH" \
    -u $UPDATE_PARA

# 5. 训练结束提醒
echo "训练命令已正常结束。"
