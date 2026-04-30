# 1. 环境与显存优化设置
# 解决 libgomp 报错
export OMP_NUM_THREADS=1
# 解决显存碎片化导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 路径设置 (请确保路径与你服务器一致)
CONFIG="configs/dfine/dfine_hgnetv2_s_dronevehicle.yml"
OUTPUT_DIR="./output/dronevehicle_s_obb_fusion_v1"
RESUME_PATH="./output/dronevehicle_s_obb_fusion_v1/last.pth"

UPDATES="checkpoint=weights_1/dfine_s_coco.pth"

echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"

# 4. 执行训练命令
# --use-amp: 开启混合精度训练，节省显存并加速
# -u: 覆盖配置文件中的设置
python train.py \
    -c $CONFIG \
    --output-dir $OUTPUT_DIR \
    --use-amp \
    --seed 42 \
    -u $UPDATES \
    -r $RESUME_PATH

# 5. 训练结束提醒
if [ $? -eq 0 ]; then
    echo "训练成功完成！"
else
    echo "训练意外中断，请检查日志。"
fi