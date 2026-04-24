import os
import sys
import torch
import argparse
from pprint import pprint

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def main(args) -> None:
    """main"""
    # 初始化分布式环境
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scratch or resume or tuning at one time"

    # 1. 解析配置参数
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if v is not None and k not in ['update', 'config']})
    
    # 2. 实例化配置对象 cfg (修复 NameError)
    cfg = YAMLConfig(args.config, **update_dict)

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    # 3. 初始化 Solver 并手动执行 Setup (修复 AttributeError)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup() 

    # 4. ================== 双流权重手动加载逻辑 ==================
    # 填入你之前训练好的 D-Fire 权重路径
    custom_checkpoint_path = './output/dfine_hgnetv2_n_dfire_160e/best_stg2.pth' 

    if os.path.exists(custom_checkpoint_path) and not args.resume:
        if safe_get_rank() == 0:
            print(f"检测到架构变更，仅映射 Backbone 权重: {custom_checkpoint_path}")
        
        checkpoint = torch.load(custom_checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        model = dist_utils.de_parallel(solver.model)
        current_model_dict = model.state_dict()
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # ================= 核心修改：仅匹配 backbone 前缀 =================
            if k.startswith('backbone.'):
                # 1. 映射到 RGB 骨干网络
                if k in current_model_dict:
                    new_state_dict[k] = v
                
                # 2. 映射到 IR 骨干网络
                ir_key = k.replace('backbone.', 'backbone_ir.')
                if ir_key in current_model_dict:
                    new_state_dict[ir_key] = v
            # ===============================================================
            # 注意：这里我们故意跳过了 encoder 和 decoder，因为结构已变
                    
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if safe_get_rank() == 0:
            print(f"成功加载双流 Backbone 权重。")
            print(f"由于架构升级，Encoder/Decoder/Fusion 层已随机初始化并准备学习新尺度特征。")

    # 5. 开始评估或训练
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 优先级 0 
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")
    parser.add_argument("--test-only", action="store_true", default=False)

    # 优先级 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # 环境参数
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank")

    args = parser.parse_args()
    main(args)