import os
import csv
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "class_names" not in checkpoint:
        raise KeyError(
            "checkpoint 中没有 class_names。请确认你使用的是 train_pytorch.py 保存的 best_checkpoint.pth。"
        )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes)

    state_dict = checkpoint["model_state_dict"]
    state_dict = strip_module_prefix(state_dict)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def find_one_image_per_class(test_dir, mode="first", seed=42):
    test_dir = Path(test_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir does not exist: {test_dir}")

    random.seed(seed)

    class_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir()])

    selected = []

    for class_dir in class_dirs:
        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        images = sorted(images)

        if len(images) == 0:
            print(f"[Warning] no image found in class: {class_dir.name}")
            continue

        if mode == "random":
            img_path = random.choice(images)
        else:
            img_path = images[0]

        selected.append((class_dir.name, img_path))

    return selected


@torch.no_grad()
def predict_one_image(model, image_path, transform, class_names, device, topk=5):
    pil_image = Image.open(image_path).convert("RGB")
    x = transform(pil_image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    topk = min(topk, len(class_names))
    scores, indices = torch.topk(probs, k=topk)

    results = []
    for score, idx in zip(scores, indices):
        cls_name = class_names[idx.item()]
        results.append((cls_name, float(score.item())))

    pred_class = results[0][0]
    pred_score = results[0][1]

    return pil_image, pred_class, pred_score, results


def save_visualization(
    image,
    true_class,
    pred_class,
    pred_score,
    topk_results,
    save_path
):
    is_correct = true_class == pred_class
    status = "Correct" if is_correct else "Wrong"

    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.axis("off")

    title = f"GT: {true_class}\nPred: {pred_class} ({pred_score:.4f}) | {status}"
    plt.title(title, fontsize=13)

    text_lines = ["Top-k:"]
    for cls_name, score in topk_results:
        text_lines.append(f"{cls_name}: {score:.4f}")

    text = "\n".join(text_lines)

    plt.gcf().text(
        0.02,
        0.02,
        text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def safe_name(name):
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/test",
        help="test folder, each subfolder is one class"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_pytorch/best_checkpoint.pth",
        help="path to best checkpoint"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="vis_one_per_class",
        help="folder to save visualization images"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="input image size"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="top-k predictions shown on each image"
    )

    parser.add_argument(
        "--select",
        type=str,
        default="first",
        choices=["first", "random"],
        help="select first image or random image from each class"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = load_model(args.checkpoint, device)
    transform = get_transform(args.size)

    selected_images = find_one_image_per_class(
        test_dir=args.test_dir,
        mode=args.select,
        seed=args.seed
    )

    print(f"Found {len(selected_images)} classes with valid images.")

    summary_path = out_dir / "summary.csv"

    rows = []

    for true_class, img_path in selected_images:
        image, pred_class, pred_score, topk_results = predict_one_image(
            model=model,
            image_path=img_path,
            transform=transform,
            class_names=class_names,
            device=device,
            topk=args.topk
        )

        correct = true_class == pred_class

        save_path = out_dir / f"{safe_name(true_class)}__pred_{safe_name(pred_class)}.png"

        save_visualization(
            image=image,
            true_class=true_class,
            pred_class=pred_class,
            pred_score=pred_score,
            topk_results=topk_results,
            save_path=save_path
        )

        rows.append({
            "true_class": true_class,
            "image_path": str(img_path),
            "pred_class": pred_class,
            "pred_score": pred_score,
            "correct": correct,
            "save_path": str(save_path)
        })

        print(
            f"[{true_class}] -> pred: {pred_class}, "
            f"score: {pred_score:.4f}, "
            f"correct: {correct}"
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "true_class",
                "image_path",
                "pred_class",
                "pred_score",
                "correct",
                "save_path"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved visualizations to: {out_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()