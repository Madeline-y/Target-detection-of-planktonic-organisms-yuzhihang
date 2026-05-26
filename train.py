import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="model_pytorch")

    parser.add_argument("--classes", type=int, default=35)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")

    parser.add_argument("--patience", type=int, default=20)

    return parser.parse_args()


def build_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tf, val_tf


def build_model(num_classes, pretrained=True, freeze_backbone=False):
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def compute_class_weights(dataset, num_classes, device):
    labels = [label for _, label in dataset.samples]
    counts = torch.zeros(num_classes, dtype=torch.float)

    for label in labels:
        counts[label] += 1

    weights = counts.sum() / (num_classes * counts)
    weights = weights.to(device)

    print("Class counts:")
    print(counts.tolist())

    print("Class weights:")
    print(weights.detach().cpu().tolist())

    return weights


def build_weighted_sampler(dataset, num_classes):
    labels = [label for _, label in dataset.samples]
    counts = torch.zeros(num_classes, dtype=torch.float)

    for label in labels:
        counts[label] += 1

    class_weights = 1.0 / counts
    sample_weights = [class_weights[label].item() for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, acc, macro_f1, balanced_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Eval", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, acc, macro_f1, balanced_acc, all_labels, all_preds


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, class_names, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "class_names": class_names,
        "args": vars(args)
    }, path)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.data_root) / "train"
    val_dir = Path(args.data_root) / "validation"
    test_dir = Path(args.data_root) / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Cannot find train dir: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Cannot find validation dir: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_tf, val_tf = build_transforms(args.size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    has_test = test_dir.exists()
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf) if has_test else None

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Detected classes:", num_classes)
    print("Class names:", class_names)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))
    if has_test:
        print("Test images:", len(test_dataset))

    if num_classes != args.classes:
        raise ValueError(f"--classes={args.classes}, but dataset has {num_classes} classes.")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(train_dataset, num_classes)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    if has_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
    else:
        test_loader = None

    model = build_model(
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    if args.use_class_weight:
        class_weights = compute_class_weights(train_dataset, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_metric = 0.0
    best_epoch = 0
    bad_epochs = 0

    log_path = output_dir / "train_log.csv"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,train_acc,train_macro_f1,train_balanced_acc,"
            "val_loss,val_acc,val_macro_f1,val_balanced_acc,lr\n"
        )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc, train_macro_f1, train_balanced_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_macro_f1, val_balanced_acc, val_labels, val_preds = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Train | loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"macro_f1={train_macro_f1:.4f} "
            f"balanced_acc={train_balanced_acc:.4f}"
        )

        print(
            f"Val   | loss={val_loss:.4f} "
            f"acc={val_acc:.4f} "
            f"macro_f1={val_macro_f1:.4f} "
            f"balanced_acc={val_balanced_acc:.4f}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_acc:.6f},{train_macro_f1:.6f},{train_balanced_acc:.6f},"
                f"{val_loss:.6f},{val_acc:.6f},{val_macro_f1:.6f},{val_balanced_acc:.6f},{lr_now:.8f}\n"
            )

        current_metric = val_macro_f1

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0

            save_checkpoint(
                output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                class_names,
                args
            )

            print(f"Saved best checkpoint. Best val macro-F1 = {best_metric:.4f}")
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_metric:.4f}")

    best_ckpt = torch.load(output_dir / "best_checkpoint.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if test_loader is not None:
        print("\nTesting best checkpoint...")

        test_loss, test_acc, test_macro_f1, test_balanced_acc, test_labels, test_preds = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Test | loss={test_loss:.4f} "
            f"acc={test_acc:.4f} "
            f"macro_f1={test_macro_f1:.4f} "
            f"balanced_acc={test_balanced_acc:.4f}"
        )

        report = classification_report(
            test_labels,
            test_preds,
            target_names=class_names,
            digits=4
        )

        cm = confusion_matrix(test_labels, test_preds)

        with open(output_dir / "test_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))

        print(report)


if __name__ == "__main__":
    main()