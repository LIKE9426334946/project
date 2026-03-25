from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset
import numpy as np

CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3"]
#! 用于可视化分割结果
ID2COLOR = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}


class SegDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None, names: List[str] | None = None):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / split / "imgs"
        self.mask_dir = self.root / split / "masks"
        self.transform = transform

        if names is None:
            self.names = sorted([p.name for p in self.image_dir.glob("*.png")])
        else:
            self.names = names

        if len(self.names) == 0:
            raise RuntimeError(f"No png files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image_path = self.image_dir / name
        mask_path = self.mask_dir / name
    
        image = Image.open(image_path).convert("RGB")
    
        # ====== 关键修改 ======
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
    
        # 白色(255,255,255) -> 1
        # 黑色(0,0,0) -> 0
        mask = (mask_rgb[:, :, 0] == 255).astype(np.uint8)
    
        mask = Image.fromarray(mask)
    
        if self.transform is not None:
            image, mask = self.transform(image, mask)
    
        return {
            "image": image,
            "mask": mask.long(),
            "name": Path(name).stem,
        }
