# SIF3D RAG Endpoint

当前 SIF3D RAG 版本的纯源码快照，基于 [SIF3D-master](https://github.com/kjle6/SIF3D-master)。保留本地当前版本的模型、数据读取、训练、评估、可视化和测试代码，包括尚未提交的 endpoint pose bank 实现。

本仓库不包含实验结果、训练日志、环境目录或环境快照、数据集、数据缓存、MotionBank、模型权重及旧 Git 历史。`requirements.txt` 仅是依赖清单。源码保留原有路径默认值；运行时需通过命令行传入自己的路径。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `model/sif3d_mod.py` | 当前 RAG 与 SIF3D 集成入口 |
| `model/multimodal_rag.py` | 多模态检索、MotionBank、原型编码与融合 |
| `model/endpoint_pose_bank.py` | 单帧终点姿态库及对应 refiner |
| `config/config.py` | 模型、数据及 RAG 命令行参数 |
| `scripts/prepare_rag_split.py` | 准备 RAG train/valid/test 数据划分 |
| `scripts/build_motion_bank.py` | 构建未来序列 MotionBank |
| `scripts/train_retriever.py` | 训练检索器并重建 bank keys |
| `scripts/cache_rag_baseline.py` | 缓存冻结 SIF3D 的预测 |
| `scripts/train_rag_refiner.py` | 训练冻结主干后的 RAG 融合模块 |
| `scripts/build_endpoint_pose_bank.py` | 构建终点姿态库 |
| `scripts/train_endpoint_pose_rag.py` | 训练终点姿态检索与融合模块 |
| `scripts/train_endpoint_adapter.py` | 训练终点原型 adapter |
| `eval.py` | 完整模型评估 |
| `tests/` | RAG、终点姿态库及 gaze 数据读取测试 |

## 依赖与外部资源

使用与项目兼容的 Python、PyTorch/CUDA 安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

PyTorch3D 等依赖需要与所用 PyTorch/CUDA 版本匹配。完整训练/评估还需要自行准备 GIMO 数据及对应缓存、SMPL-X 模型、VPoser 权重、SIF3D/轨迹预测权重，以及与 RAG checkpoint 匹配的训练集 MotionBank。`vposer_v1_0/` 仅保留 Python 源码及 INI 配置，不提供权重。

## 运行示例

以下评估命令演示 `adaptive_terminal` 版本的参数组合。请将 `/path/to/...` 替换为自己的资源路径，checkpoint、bank 和 refiner 类型必须匹配：

```bash
python eval.py \
  --use_rag --rag_train_stage refiner \
  --rag_bank /path/to/trained.bank.pt \
  --rag_refiner_type adaptive_terminal --rag_refiner_hidden_dim 0 \
  --rag_terminal_max_trust 1.2 \
  --rag_allow_same_sequence --rag_prefer_same_scene \
  --rag_top_k 3 --rag_endpoint_frames 4 \
  --load_model_dir /path/to/rag_checkpoint.pth \
  --traj_ckpts /path/to/trajectory_checkpoint.pth \
  --dataroot /path/to/prepared_gimo \
  --dataset_csv dataset --folder cache \
  --sample_points 32768 --occ --batch_size 1 --num_workers 0 \
  --smplx_path /path/to/smplx --vposer_path /path/to/vposer \
  --output_path runs/evaluation
```

单帧终点姿态库使用 `pose_endpoint*` refiner，不能与未来序列 MotionBank 互换。各训练脚本的参数可通过 `python scripts/<script>.py --help` 查看。

## 测试

```bash
python -m pytest tests -q
```

测试使用合成输入和临时文件，不要求下载训练数据或模型权重。完整训练和真实数据评估需要前述外部资源。
