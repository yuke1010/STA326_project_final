# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from types import SimpleNamespace
from models.utils import build_model


class TestImageDataset(Dataset):
    def __init__(self, img_dir, sample_csv):
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(sample_csv)
        self.ids = self.df["id"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_dir / img_id
        image = Image.open(img_path).convert("RGB")
        return image, img_id


def build_tta_transforms(img_size=512):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def base_ops(extra_ops):
        return transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(img_size),
            *extra_ops,
            transforms.ToTensor(),
            normalize,
        ])

    tta = {
        "orig": base_ops([]),
        "hflip": base_ops([transforms.RandomHorizontalFlip(p=1.0)]),
        "vflip": base_ops([transforms.RandomVerticalFlip(p=1.0)]),
        "hvflip": base_ops([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
        ]),
    }
    return tta


def collate_pil(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/final project/data/raw")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/final project/experiments/v07_original_dinov3_tta")
    parser.add_argument("--model", type=str, default="dinov3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model_args = SimpleNamespace(model=args.model)
    model = build_model(model_args, device=device, num_classes=2)

    print("Loading checkpoint:", args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img_dir = Path(args.data_path) / "test_images" / "test_images"
    sample_csv = Path(args.data_path) / "sample_submission.csv"

    dataset = TestImageDataset(img_dir, sample_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
        pin_memory=True,
    )

    tta_transforms = build_tta_transforms(img_size=512)
    print("TTA views:", list(tta_transforms.keys()))

    all_ids = []
    all_prob_sum = None
    view_probs = {}

    with torch.no_grad():
        for view_name, tfm in tta_transforms.items():
            print(f"\nRunning TTA view: {view_name}")
            probs_this_view = []
            ids_this_view = []

            for pil_images, img_ids in tqdm(loader):
                x = torch.stack([tfm(im) for im in pil_images], dim=0).to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(x)

                prob = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu()

                probs_this_view.append(prob)
                ids_this_view.extend(img_ids)

            probs_this_view = torch.cat(probs_this_view).numpy()

            if not all_ids:
                all_ids = ids_this_view
            else:
                assert all_ids == ids_this_view, "Image order mismatch among TTA views."

            view_probs[view_name] = probs_this_view

            if all_prob_sum is None:
                all_prob_sum = probs_this_view.copy()
            else:
                all_prob_sum += probs_this_view

    mean_prob = all_prob_sum / len(tta_transforms)
    pred = (mean_prob >= args.threshold).astype(int)

    prob_df = pd.DataFrame({"id": all_ids, "prob": mean_prob})
    for view_name, p in view_probs.items():
        prob_df[f"prob_{view_name}"] = p

    prob_df.to_csv(output_dir / "test_prob_tta.csv", index=False)

    sub_df = pd.DataFrame({"id": all_ids, "label": pred})
    sub_df.to_csv(output_dir / "submission.csv", index=False)

    print("\nSaved:")
    print(output_dir / "test_prob_tta.csv")
    print(output_dir / "submission.csv")

    print("\nLabel counts:")
    print(sub_df["label"].value_counts())
    print(sub_df["label"].value_counts(normalize=True))

    print("\nProbability summary:")
    print(prob_df["prob"].describe())


if __name__ == "__main__":
    main()
