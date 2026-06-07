# -*- coding: utf-8 -*-
"""
最终融合 Python 脚本
v13 + Qwen 校准 0.10，权重 0.25（q010_w025）
"""

import numpy as np
import pandas as pd
from pathlib import Path

def calibrate(prob, th):
    eps = 1e-6
    p = prob.clip(eps, 1 - eps)
    logit = np.log(p / (1 - p))
    offset = np.log(th / (1 - th))
    return 1 / (1 + np.exp(-(logit - offset)))

def main():
    qwen_root = Path("/data/final project/experiments/vlm_teacher_qwen3")

    # 三个传统模型概率
    dino = pd.read_csv("/data/final project/experiments/v07_original_dinov3_tta/test_prob_tta.csv")[["id","prob"]].rename(columns={"prob":"prob_dino"})
    base = pd.read_csv("/data/final project/experiments/v10_convnextv2_base_frozen_tta/test_prob.csv")[["id","prob"]].rename(columns={"prob":"prob_base"})
    large = pd.read_csv("/data/final project/experiments/v12_convnextv2_large_frozen_tta/test_prob.csv")[["id","prob"]].rename(columns={"prob":"prob_large"})

    # Qwen 概率
    qwen = pd.read_csv(qwen_root / "qwen3_32b_thinking_noreason_test_scores.csv")[["id","prob_vlm"]].rename(columns={"prob_vlm":"prob_qwen"}).dropna()

    # v13 submission 作为对比基准
    v13_sub = pd.read_csv("/data/final project/submissions/submission_v13_blend_d080_b015_l005.csv")

    df = dino.merge(base, on="id").merge(large, on="id").merge(qwen, on="id")

    # v13 概率
    df["prob_v13"] = 0.80 * df["prob_dino"] + 0.15 * df["prob_base"] + 0.05 * df["prob_large"]

    # Qwen 校准 0.10
    df["prob_qwen_cal_t010"] = calibrate(df["prob_qwen"], 0.10)

    # 最终最优权重 q010_w025
    wqwen = 0.25
    tmp = df.copy()
    tmp["prob_blend"] = (1 - wqwen) * tmp["prob_v13"] + wqwen * tmp["prob_qwen_cal_t010"]
    tmp["label"] = (tmp["prob_blend"] >= 0.5).astype(int)

    out_dir = Path("/data/final project/experiments/v40_v13_qwen_q010_w025")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp.to_csv(out_dir / "test_prob.csv", index=False)
    tmp[["id","label"]].to_csv(out_dir / "submission.csv", index=False)

    comp = v13_sub.merge(tmp[["id","label","prob_v13","prob_qwen","prob_qwen_cal_t010","prob_blend"]],
                         on="id", suffixes=("_v13","_new"))
    changed = comp[comp["label_v13"] != comp["label_new"]]

    print("最终融合完成：q010_w025")
    print("保存路径:")
    print("submission.csv:", out_dir / "submission.csv")
    print("test_prob.csv:", out_dir / "test_prob.csv")
    print("不同于 v13 的数量:", len(changed))
    print(changed.head(20)[["id","label_v13","label_new","prob_v13","prob_qwen","prob_qwen_cal_t010","prob_blend"]])

if __name__ == "__main__":
    main()