from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3"]
#! 用于可视化分割结果
ID2COLOR = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

class SegDataset(Dataset):
    def __init__(self, root, split_file, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        if split in ["train", "val"]:
            self.image_dir = self.root / "training" / "images2"
            self.mask_dir = self.root / "training" / "class_si"
        elif split == "test":
            self.image_dir = self.root / "test" / "images2"
            self.mask_dir = self.root / "test" / "class_si"
        else:
            raise ValueError(f"Unsupported split: {split}")

        with open(split_file, "r", encoding="utf-8") as f:
            self.names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image_path = self.image_dir / name
        mask_path = self.mask_dir / name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)  # 单通道类别图

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask.long(),
            "name": Path(name).stem,
        }
