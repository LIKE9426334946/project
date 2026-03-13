import argparse
import random
from pathlib import Path



def make_split(data_root: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    root = Path(data_root)
    train_img_dir = root / "train" / "imgs"
    train_mask_dir = root / "train" / "masks"
    val_img_dir = root / "val" / "imgs"
    val_mask_dir = root / "val" / "masks"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = sorted([p.name for p in train_img_dir.glob("*.png") if (train_mask_dir / p.name).exists()])
    val_names = sorted([p.name for p in val_img_dir.glob("*.png") if (val_mask_dir / p.name).exists()])

    if len(train_names) == 0:
        raise RuntimeError("No matched png pairs found under train/imgs and train/masks")
    if len(val_names) == 0:
        raise RuntimeError("No matched png pairs found under val/imgs and val/masks")

    rnd = random.Random(seed)
    rnd.shuffle(train_names)
    test_count = max(1, int(len(train_names) * test_ratio))
    test_names = sorted(train_names[:test_count])
    real_train_names = sorted(train_names[test_count:])

    for split_name, names in {
        "train": real_train_names,
        "val": val_names,
        "test": test_names,
    }.items():
        with open(output_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(names))

    print(f"Train split source size: {len(train_names)}")
    print(f"Generated train: {len(real_train_names)}, val: {len(val_names)}, test: {len(test_names)}")
    print(f"Split files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/dataset")
    parser.add_argument("--output_dir", type=str, default="runs/splits")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_split(args.data_root, args.output_dir, test_ratio=args.test_ratio, seed=args.seed)
