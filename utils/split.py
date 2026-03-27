import argparse
import random
from pathlib import Path


def make_split(data_root: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
    root = Path(data_root)

    train_img_dir = root / "training" / "images2"
    train_mask_dir = root / "training" / "class_si"
    test_img_dir = root / "test" / "images2"
    test_mask_dir = root / "test" / "class_si"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = sorted([
        p.name for p in train_img_dir.glob("*.bmp")
        if (train_mask_dir / p.name.replace(".bmp", ".png")).exists()
    ])
    test_names = sorted([
        p.name for p in test_img_dir.glob("*.bmp")
        if (test_mask_dir / p.name.replace(".bmp", ".png")).exists()
    ])

    if len(train_names) == 0:
        raise RuntimeError("No matched png pairs found under training/images2 and training/class_si")
    if len(test_names) == 0:
        raise RuntimeError("No matched png pairs found under test/images2 and test/class_si")

    rnd = random.Random(seed)
    rnd.shuffle(train_names)

    val_count = max(1, int(len(train_names) * val_ratio))
    val_names = sorted(train_names[:val_count])
    real_train_names = sorted(train_names[val_count:])

    for split_name, names in {
        "train": real_train_names,
        "val": val_names,
        "test": test_names,
    }.items():
        with open(output_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(names))

    print(f"Training source size: {len(train_names)}")
    print(f"Generated train: {len(real_train_names)}, val: {len(val_names)}, test: {len(test_names)}")
    print(f"Split files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/data1600")
    parser.add_argument("--output_dir", type=str, default="runs/splits")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_split(args.data_root, args.output_dir, val_ratio=args.val_ratio, seed=args.seed)
