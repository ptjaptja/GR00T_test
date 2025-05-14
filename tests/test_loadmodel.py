import os
import torch
from pathlib import Path
# from transformers import AutoConfig, AutoModel, AutoTokenizer
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


# MODEL_PATH = "nvidia/GR00T-N1-2B"
MODEL_PATH = "/data/data2/models/GR00T-N1-2B"
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = "gr1"


device = "cuda" if torch.cuda.is_available() else "cpu"

def validate_model_dir(model_dir: str) -> bool:
    # 1. 转为绝对路径
    # model_dir = os.path.abspath(model_dir)                                       # :contentReference[oaicite:9]{index=9}
    # 2. 检查目录存在
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    # 3. 检查核心文件
    required = ["config.json", "model.safetensors"]                               # 根据实际情况调整
    present = os.listdir(model_dir)
    for fname in required:
        if fname not in present:
            raise FileNotFoundError(f"缺少文件: {fname} in {model_dir}")
    # 4. 尝试加载配置与模型
    data_config = DATA_CONFIG_MAP["gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_dir,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    # AutoTokenizer.from_pretrained(model_dir)                                       # :contentReference[oaicite:12]{index=12}
    return True

# 调用示例
try:
    validate_model_dir(MODEL_PATH)
    print("本地模型加载路径验证通过。")
except Exception as e:
    print(f"验证失败：{e}")