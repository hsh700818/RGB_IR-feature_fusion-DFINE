# 1. 环境与显存优化设置
# 解决 libgomp 报错
export OMP_NUM_THREADS=1
# 解决显存碎片化导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 路径设置 (请确保路径与你服务器一致)
CONFIG="configs/dfine/dfine_hgnetv2_n_flame2.yml"
OUTPUT_DIR="./output/flame2_add_moudule_80e"
# RESUME_PATH="./output/flame2_160e/last.pth"

# 3. 训练参数定义
# epochs=160: 总轮数
# stop_epoch=148: 在148轮关闭增强，触发D-FINE特有的FDR细化
# total_batch_size=16: 训练批次 (双流模型显存压力大，16比较稳)
# val_batch_size=4: 验证批次 [关键：调低验证批次以防止第10轮验证时OOM]
UPDATES="epochs=80 \
train_dataloader.collate_fn.stop_epoch=68 \
train_dataloader.total_batch_size=16 \
val_dataloader.total_batch_size=4 \
checkpoint='./output/flame2_160e/last.pth'"

echo "==================== 开始 D-FINE 双流训练 ===================="
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
    # -r $RESUME_PATH

# 5. 训练结束提醒
if [ $? -eq 0 ]; then
    echo "训练成功完成！"
else
    echo "训练意外中断，请检查日志。"
fi