# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import timm


class ImageClsDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img = Image.open(self.img_dir / img_id).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_id

        label = int(row["label"])
        return img, label


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, test_tf


def build_tta_transforms(img_size):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def make(extra):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            *extra,
            transforms.ToTensor(),
            norm,
        ])

    return {
        "orig": make([]),
        "hflip": make([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": make([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": make([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
    }


def freeze_backbone_train_head(model):
    for p in model.parameters():
        p.requires_grad = False

    # Most timm classifiers expose reset_classifier and get_classifier.
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    return model


def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    all_probs, all_labels = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": float(total_loss / max(total_n, 1)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def predict(model, loader, device, use_amp=True):
    model.eval()
    ids, probs = [], []

    with torch.no_grad():
        for x, img_ids in tqdm(loader, desc="Test"):
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits = model(x)

            prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

            ids.extend(list(img_ids))
            probs.append(prob)

    return ids, np.concatenate(probs)


def predict_tta(model, test_df, img_dir, img_size, batch_size, num_workers, device):
    tta_tfs = build_tta_transforms(img_size)
    view_probs = {}
    ids_ref = None

    for view_name, tfm in tta_tfs.items():
        ds = ImageClsDataset(test_df, img_dir, transform=tfm, is_test=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        ids, probs = predict(model, loader, device)

        if ids_ref is None:
            ids_ref = ids
        else:
            assert ids_ref == ids, "TTA order mismatch."

        view_probs[view_name] = probs

    mean_prob = np.mean(list(view_probs.values()), axis=0)
    return ids_ref, mean_prob, view_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/final project/data/raw")
    parser.add_argument("--output_root", type=str, default="/data/final project/experiments")
    parser.add_argument("--exp_name", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="convnextv2_base")
    parser.add_argument("--img_size", type=int, default=384)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None)

    args = parser.parse_args()

    exp_dir = Path(args.output_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("Experiment:", args.exp_name)
    print("Args:", vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data_root = Path(args.data_path)
    train_csv = data_root / "train_labels.csv"
    sample_csv = data_root / "sample_submission.csv"
    train_img_dir = data_root / "train_images" / "train_images"
    test_img_dir = data_root / "test_images" / "test_images"

    df = pd.read_csv(train_csv)
    test_df = pd.read_csv(sample_csv)

    tr_df, val_df = train_test_split(
        df,
        test_size=0.10,
        random_state=args.seed,
        stratify=df["label"]
    )

    print("Train:", len(tr_df), "Val:", len(val_df), "Test:", len(test_df))

    train_tf, val_tf = build_transforms(args.img_size)

    train_ds = ImageClsDataset(tr_df, train_img_dir, train_tf, is_test=False)
    val_ds = ImageClsDataset(val_df, train_img_dir, val_tf, is_test=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.test_bs, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print("Creating model:", args.model_name)

    if args.pretrained_path:
        print("Loading local pretrained weight via timm:", args.pretrained_path)

        # Important:
        # Use timm's own pretrained loader instead of manually calling load_state_dict.
        # Some official checkpoints use Facebook key names such as:
        #   downsample_layers.0.0.weight
        # while timm models expect:
        #   stem.0.weight
        #   stages.0.blocks.0.conv_dw.weight
        # timm's internal checkpoint_filter_fn handles this conversion.
        model = timm.create_model(
            args.model_name,
            pretrained=True,
            num_classes=2,
            pretrained_cfg_overlay={"file": args.pretrained_path},
        )
    else:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=2)

    model = freeze_backbone_train_head(model)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", trainable)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = -1.0
    best_epoch = 0
    no_improve = 0
    rows = []

    best_path = exp_dir / "best_head.pth"
    last_path = exp_dir / "last_head.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n, total_correct = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(exp_dir / "metrics.csv", index=False)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        head_state = model.get_classifier().state_dict()
        torch.save({"head_state_dict": head_state, "epoch": epoch, "metrics": val_metrics, "args": vars(args)}, last_path)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"head_state_dict": head_state, "epoch": epoch, "metrics": val_metrics, "args": vars(args)}, best_path)
            print("New best by val_acc:", best_acc, "epoch:", epoch)
        else:
            no_improve += 1
            print(f"No improvement: {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print("Early stopping.")
                break

    # Load best head.
    ckpt = torch.load(best_path, map_location=device)
    model.get_classifier().load_state_dict(ckpt["head_state_dict"])
    model.eval()

    if args.no_tta:
        test_tf = build_tta_transforms(args.img_size)["orig"]
        test_ds = ImageClsDataset(test_df, test_img_dir, test_tf, is_test=True)
        test_loader = DataLoader(test_ds, batch_size=args.test_bs, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        ids, prob = predict(model, test_loader, device)
        view_probs = {}
    else:
        ids, prob, view_probs = predict_tta(
            model, test_df, test_img_dir, args.img_size,
            args.test_bs, args.num_workers, device
        )

    out_prob = pd.DataFrame({"id": ids, "prob": prob})
    for k, v in view_probs.items():
        out_prob[f"prob_{k}"] = v

    out_prob.to_csv(exp_dir / "test_prob.csv", index=False)

    sub = pd.DataFrame({
        "id": ids,
        "label": (prob >= 0.5).astype(int),
    })
    sub.to_csv(exp_dir / "submission.csv", index=False)

    summary = {
        "exp_name": args.exp_name,
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "submission_path": str(exp_dir / "submission.csv"),
        "test_prob_path": str(exp_dir / "test_prob.csv"),
    }
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Label counts:")
    print(sub["label"].value_counts())
    print(sub["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
