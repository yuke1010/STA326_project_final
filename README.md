# STA326_project_final

## Project Overview

This repository contains the final project for STA326. The project implements an image classification pipeline based on multiple models, including **DINOV3**, **ConvNeXtV2**, and a **Qwen VLM Teacher** model. The final submission is produced through probability-level model blending.

The repository includes training scripts, inference scripts, model blending scripts, final prediction files, and the final submission file. The purpose of this README is to explain how to reproduce the final test results.

## Repository Structure

```text
STA326_project_final/
├── experiments/
│   └── v40_v13_qwen_q010_w025/
│       ├── submission.csv
│       └── test_prob.csv
├── external/
│   └── teammate_repo/
│       └── final_v13_q010_w025_blend.py
├── hhp_main.py
├── infer_tta_dinov3.py
├── train_timm_frozen.py
├── run_vlm_thinking_noreason.py
├── final_inventory.txt
└── README.md
```

## Environment Requirements

The recommended Python version is:

```bash
Python 3.11+
```

Install the required packages:

```bash
pip install torch torchvision timm pandas numpy scikit-learn tqdm opencv-python einops
```

If GPU acceleration is available, please make sure that CUDA and the corresponding PyTorch version are correctly installed.

## Data Preparation

Please place the dataset provided by the course into the `data/` directory.

The expected structure is:

```text
data/
├── train/
├── test/
└── sample_submission.csv
```

The `train/` directory should contain the training images, and the `test/` directory should contain the test images.

## Reproducing the Final Result

The final submission file is located at:

```text
experiments/v40_v13_qwen_q010_w025/submission.csv
```

To reproduce the final result, follow the steps below.

### Step 1: Train DINOV3 Model

Run the DINOV3 training script:

```bash
python hhp_main.py
```

This script trains the DINOV3-based image classification model.

### Step 2: Train ConvNeXtV2 Model

Run the ConvNeXtV2 frozen training script:

```bash
python train_timm_frozen.py
```

This script trains the ConvNeXtV2-based classifier with frozen backbone settings.

### Step 3: Run Qwen VLM Teacher Inference

Run the Qwen VLM Teacher inference script:

```bash
python run_vlm_thinking_noreason.py
```

This script generates teacher-model predictions for the test set.

### Step 4: Generate DINOV3 Test Probabilities

Run the DINOV3 test-time augmentation inference script:

```bash
python infer_tta_dinov3.py
```

The generated test probability file is saved as:

```text
experiments/v40_v13_qwen_q010_w025/test_prob.csv
```

### Step 5: Run Final Model Blending

Run the final blending script:

```bash
python external/teammate_repo/final_v13_q010_w025_blend.py
```

The final blended submission file will be saved as:

```text
experiments/v40_v13_qwen_q010_w025/submission.csv
```

## Final Output

The final submission file used for evaluation is:

```text
experiments/v40_v13_qwen_q010_w025/submission.csv
```

The corresponding test probability file is:

```text
experiments/v40_v13_qwen_q010_w025/test_prob.csv
```

## Notes

* `submission.csv` is the final file for submission.
* `test_prob.csv` stores the model probability outputs used for final blending.
* `final_v13_q010_w025_blend.py` is the final ensemble script.
* `final_inventory.txt` records the important files included in the final version.
* To reproduce the result, please make sure that the dataset path and output path are consistent with the repository structure above.

## Author

Yuke1010

## Repository Link

https://github.com/yuke1010/STA326_project_final
