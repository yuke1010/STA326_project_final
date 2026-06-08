# STA326_project_final

## 仓库概览

本仓库为 STA326 课程最终项目代码仓库，包含基于 **DINOV3**、**ConvNeXtV2** 与 **Qwen VLM Teacher** 的图像分类实验流程。项目主要包括模型训练、测试集推理、测试概率生成以及最终融合提交文件生成。

本仓库保留了最终实验版本所需的核心代码、最终概率文件、融合脚本和提交文件，便于复现实验结果。

## 目录结构

```
STA326_project_final/
├── experiments/
│   └── v40_v13_qwen_q010_w025/
│   │   ├── submission.csv
│   │   └── test_prob.csv
│   ├── submission_v13_blend_d080_b015_l005.csv
│   ├── v07_original_dinov3_tta_test_prob.csv
│   ├── v10_convnextv2_base_frozen_tta_test_prob.csv
│   └── v12_convnextv2_large_frozen_tta_test_prob.csv
├── external/
│   └── teammate_repo/
│       ├── final_v13_q010_w025_blend.py
│       ├── hhp_main.py
│       ├── infer_tta_dinov3.py
│       ├── run_vlm_thinking_noreason.py
│       └── train_timm_frozen.py
├── final_inventory.txt
└── README.md
```

## 环境要求

建议使用以下 Python 版本：

```
Python 3.11+
```

安装主要依赖：

```
pip install torch torchvision timm pandas numpy scikit-learn tqdm opencv-python einops
```

使用 GPU 推理或训练，请确保已正确安装 CUDA 与对应版本 PyTorch。

## 数据准备

请将课程提供的数据集放置在 `data/` 目录下，推荐目录结构如下：

```
data/
├── train/
├── test/
└── sample_submission.csv
```

其中：

* `train/` 用于存放训练集图像；
* `test/` 用于存放测试集图像；
* `sample_submission.csv` 为课程提供的提交文件模板。

## 模型下载

本项目最终复现实验结果所需的模型权重文件较大，因此未直接上传至 GitHub。请通过以下百度网盘链接下载模型压缩包。

模型包名称：STA326_project_final_models.tar
百度网盘链接：[点此下载](https://pan.baidu.com/s/1AIyY-OXFXy9IUFsNYDTmNQ?pwd=yxup)
提取码：yxup

下载完成后，请将模型压缩包解压到本项目根目录下。

示例命令如下：

### 假设模型包下载到了 ~/Downloads 目录
```
cd ~/Downloads
```

### 将模型文件解压到项目根目录
```
tar -xvf STA326_project_final_models.tar -C /your/project/path
```

其中 /your/project/path 请替换为你本地的项目路径，例如：

```
tar -xvf STA326_project_final_models.tar -C ~/STA326_project_final
```

解压后，模型文件目录示例如下：

```
STA326_project_final/
├── external/
│   └── teammate_repo/
│       └── reproduce_original_dinov3_e20_best.pth
├── experiments/
│   ├── convnextv2_base/
│   │   └── best_head_base.pth
│   └── convnextv2_large/
│       └── best_head_large.pth
```

请保持模型文件的原始目录结构，不要手动修改模型文件名或移动模型文件位置，否则可能导致推理脚本无法正确读取权重。

## 实验复现流程

### 1. DINOV3 模型训练

运行 DINOV3 训练脚本：

```
python hhp_main.py \
  --data_path "/data/final project/data/raw" \
  --model dinov3 \
  --epochs 20 \
  --batch_size 16 \
  --test_bs 16 \
  --num_workers 4 \
  --learning_rate 4e-5 \
  --decay 2e-2 \
  --exp_name reproduce_original_dinov3_e20
```

该脚本用于训练基于 DINOV3 的图像分类模型。


### 2. ConvNeXtV2 模型训练

#### 2.1 训练 Base 模型
运行以下命令训练 ConvNeXtV2 Base：

```bash
python train_timm_frozen.py \
  --data_path "/data/final project/data/raw" \
  --output_root "/data/final project/experiments" \
  --model_name convnextv2_base \
  --img_size 384 \
  --epochs 20 \
  --batch_size 16 \
  --test_bs 16 \
  --num_workers 4 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --patience 8 \
  --seed 35 \
  --exp_name v10_convnextv2_base_frozen_tta
```

#### 2.2 训练 Large 模型

运行以下命令训练 ConvNeXtV2 Large：

```bash
python train_timm_frozen.py \
  --data_path "/data/final project/data/raw" \
  --output_root "/data/final project/experiments" \
  --model_name convnextv2_large \
  --img_size 384 \
  --epochs 20 \
  --batch_size 8 \
  --test_bs 8 \
  --num_workers 4 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --patience 8 \
  --seed 35 \
  --exp_name v12_convnextv2_large_frozen_tta
```

### 3. Qwen VLM Teacher 推理

运行 Qwen VLM Teacher 推理脚本：

```
python run_vlm_thinking_noreason.py
```

该脚本用于生成 Qwen VLM Teacher 在测试集上的预测结果。

⚠️ 注意：
- 本实验使用的是“硅基流动”服务（SiliconFlow）的 Qwen API。
- 若其他人复现，需要在脚本开头或配置文件中设置自己的 API Key：
```python
SILICONFLOW_API_KEY = "YOUR_OWN_API_KEY"
```
- 或者修改脚本调用其他可用的 VLM API。
- 确保网络可访问该 API 服务，才能生成预测结果。

### 4. 生成 DINOV3 测试概率

运行 DINOV3 测试时增强推理脚本：

```
python infer_tta_dinov3.py
```

生成的测试概率文件保存路径为：

```
experiments/v40_v13_qwen_q010_w025/test_prob.csv
```

### 5. 最终模型融合

运行最终融合脚本：

```
python external/teammate_repo/final_v13_q010_w025_blend.py
```

融合后生成最终提交文件：

```
experiments/v40_v13_qwen_q010_w025/submission.csv
```

## 最终结果文件

最终提交文件为：

```
experiments/v40_v13_qwen_q010_w025/submission.csv
```

测试概率文件为：

```
experiments/v40_v13_qwen_q010_w025/test_prob.csv
```

## 文件说明

* `hhp_main.py`：DINOV3 模型训练脚本。
* `infer_tta_dinov3.py`：DINOV3 测试集推理与测试时增强脚本。
* `train_timm_frozen.py`：ConvNeXtV2 冻结训练脚本。
* `run_vlm_thinking_noreason.py`：Qwen VLM Teacher 推理脚本。
* `external/teammate_repo/final_v13_q010_w025_blend.py`：最终融合脚本。
* `experiments/v40_v13_qwen_q010_w025/test_prob.csv`：测试集概率输出文件。
* `experiments/v40_v13_qwen_q010_w025/submission.csv`：最终提交文件。
* `final_inventory.txt`：最终实验文件清单。

## 注意事项

* 运行代码前，请确认数据路径与脚本中的路径设置一致。
* 如更换数据目录，需要同步修改对应脚本中的输入路径。
* `submission.csv` 是最终用于提交的文件。
* `test_prob.csv` 是最终融合所需的测试概率文件。
* 为保证实验可复现，请勿随意删除 `experiments/v40_v13_qwen_q010_w025/` 目录下的文件。
* 如需重新训练模型，请确保 GPU、依赖库和数据集均已正确配置。

## 作者

Yuke1010

## 仓库链接

https://github.com/yuke1010/STA326_project_final
