import os
import random
import shutil
from pathlib import Path


def split_dataset(
    src_dir="data/zooplankton_0p5x",
    dst_dir="data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    mode="copy"
):
    random.seed(seed)

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    train_dir = dst_dir / "train"
    val_dir = dst_dir / "validation"
    test_dir = dst_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_exts = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp", ".ppm", ".pgm"
    }

    class_dirs = sorted([p for p in src_dir.iterdir() if p.is_dir()])

    print(f"Source dir: {src_dir}")
    print(f"Number of classes: {len(class_dirs)}")

    total_all = 0

    for class_dir in class_dirs:
        class_name = class_dir.name

        images = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        images = sorted(images)
        random.shuffle(images)

        n_total = len(images)
        total_all += n_total

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        split_map = {
            "train": train_imgs,
            "validation": val_imgs,
            "test": test_imgs,
        }

        for split_name, split_imgs in split_map.items():
            out_class_dir = dst_dir / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for i, img_path in enumerate(split_imgs):
                # 防止不同子目录里有同名文件
                out_name = f"{class_name}_{i:06d}{img_path.suffix.lower()}"
                out_path = out_class_dir / out_name

                if out_path.exists():
                    continue

                if mode == "copy":
                    shutil.copy2(img_path, out_path)
                elif mode == "symlink":
                    os.symlink(img_path.resolve(), out_path)
                else:
                    raise ValueError("mode must be copy or symlink")

        print(
            f"{class_name:25s} total={n_total:5d} "
            f"train={len(train_imgs):5d} "
            f"val={len(val_imgs):5d} "
            f"test={len(test_imgs):5d}"
        )

    print(f"\nTotal images found: {total_all}")

    if total_all == 0:
        print("\nError: no images found.")
        print("Please check file extensions with:")
        print("find data/zooplankton_0p5x -type f | head -30")
        print("find data/zooplankton_0p5x -type f | sed 's/.*\\.//' | sort | uniq -c")


if __name__ == "__main__":
    split_dataset(
        src_dir="data/zooplankton_0p5x",
        dst_dir="data",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        mode="copy"
    )