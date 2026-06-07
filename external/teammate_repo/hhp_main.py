# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_loader.utils import build_dataset, build_all_datasets
from models.utils import build_model, save_model, load_model
from utils import inference_and_save, validate


def get_labels_from_dataset(dataset):
    if isinstance(dataset, torch.utils.data.dataset.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices

        # 尝试多种方式获取原始标签
        if hasattr(original_dataset, "df") and "label" in original_dataset.df.columns:
            all_labels = np.array(original_dataset.df["label"].values)
            return all_labels[indices]

        if hasattr(original_dataset, "labels"):
            all_labels = np.array(original_dataset.labels)
            return all_labels[indices]

        if hasattr(original_dataset, "targets"):
            all_labels = np.array(original_dataset.targets)
            return all_labels[indices]

        # 如果原始 dataset 有 .samples 或 .imgs (如 ImageFolder)
        if hasattr(original_dataset, "samples"):
            # samples 是 (path, label) 的列表
            all_labels = np.array([label for _, label in original_dataset.samples])
            return all_labels[indices]


def make_exp_name(args):
    if args.exp_name != "":
        return args.exp_name

    return (
        f"{args.model}"
        f"_lr{args.learning_rate}"
        f"_wd{args.decay}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_seed{args.seed}"
    )


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--data_path", default="./stage2dataset")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="dinov3",
        help="Choose architecture."
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=20,
        help="Number of epochs to train. Default to 0 means no training."
    )

    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Train batch size.")
    parser.add_argument("--test_bs", type=int, default=32, help="Test batch size.")

    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=4e-5,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--decay",
        "-d",
        type=float,
        default=2e-2,
        help="Weight decay for AdamW."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader."
    )

    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Set to the model path that you want to load."
    )

    parser.add_argument("--save", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--seed", type=int, default=35)

    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience."
    )

    args = parser.parse_args()

    exp_name = make_exp_name(args)

    state = {k: v for k, v in args._get_kwargs()}
    print("Hyperparameters for the experiment")
    print(state)
    print(f"Experiment name: {exp_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True


    train_data, val_data, test_data, num_classes = build_all_datasets(args.data_path, args)


    #
    # train_sampler = build_weighted_sampler(train_data, num_classes)

    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     sampler=train_sampler,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"\n[Dataset Info] Training Set: {len(train_data)} | Validation Set: {len(val_data)}")
    print(f"[Dataset Info] Num Classes: {num_classes}\n")


    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ---------------- 2. Build and load model ----------------
    net = build_model(args, device=device, num_classes=num_classes)

    if args.load_model_path is not None:
        print(f"\n[Info] Loading weights from {args.load_model_path}...")
        net = load_model(net, args.load_model_path, device=device)

    # ---------------- 3. Training and validation ----------------
    if args.epochs > 0:
        print("----------------Training & Validation--------------------\n")

        optimizer = optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=(args.amp and device == "cuda")
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        print(
            "[Loss Info] Using CrossEntropyLoss with label smoothing=0.1\n"
        )

        best_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state = None

        os.makedirs("./checkpoints", exist_ok=True)

        best_model_path = os.path.join(
            "./checkpoints",
            f"{exp_name}_best.pth"
        )

        last_model_path = os.path.join(
            "./checkpoints",
            f"{exp_name}_last.pth"
        )

        for e in range(1, args.epochs + 1):
            # ---------------- Train ----------------
            net.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {e}/{args.epochs} [Train]")

            total_train_loss = 0.0
            train_total = 0
            train_correct = 0

            for images, labels in train_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                    logits = net(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = labels.size(0)
                total_train_loss += loss.item() * batch_size

                preds = torch.argmax(logits.detach(), dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += batch_size

                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()

            avg_train_loss = total_train_loss / train_total
            train_acc = 100.0 * train_correct / train_total

            # ---------------- Validation ----------------
            avg_val_loss, val_acc = validate(net, val_loader, args, e, device, criterion)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"-> Epoch {e}/{args.epochs} | "
                f"LR: {current_lr:.6e} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if args.save == "True":
                save_model(net, save_path=last_model_path)

            # Save best model by validation accuracy.
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = e
                epochs_no_improve = 0

                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }

                if args.save == "True":
                    print(
                        f"🌟 New best model | "
                        f"Epoch: {best_epoch} | "
                        f"Val Acc: {best_acc:.2f}% | "
                        f"Save path: {best_model_path}"
                    )
                    save_model(net, save_path=best_model_path)

            else:
                epochs_no_improve += 1
                print(
                    f"[EarlyStopping] No improvement for "
                    f"{epochs_no_improve}/{args.patience} epochs."
                )

                if epochs_no_improve >= args.patience:
                    print(
                        f"[EarlyStopping] Stop training at epoch {e}. "
                        f"Best epoch: {best_epoch}, Best Val Acc: {best_acc:.2f}%"
                    )
                    break

            print()

        print(
            f"Training completed! "
            f"Best Val Acc: {best_acc:.2f}% at epoch {best_epoch}"
        )

        if best_model_state is not None:
            net.load_state_dict(best_model_state)
            net = net.to(device)

    # ---------------- 4. Test inference ----------------
    print("\n----------------Testing--------------------")

    current_time = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("./result", exist_ok=True)

    if args.epochs == 0 and args.load_model_path:
        load_name = os.path.splitext(os.path.basename(args.load_model_path))[0]
        save_csv_path = os.path.join(
            "./result",
            f"submission_{load_name}_test_only_{current_time}.csv"
        )
    else:
        save_csv_path = os.path.join(
            "./result",
            f"submission_{exp_name}_{current_time}.csv"
        )

    print("Running inference on test set...")

    inference_and_save(
        test_loader,
        net,
        device,
        save_csv_path=save_csv_path
    )

    print(f"✅ Inference completed! Results successfully saved to {save_csv_path}\n")