import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


# custom dataset for loading sequential satellite images for ConvLSTM training
class SatelliteSequenceDataset(Dataset):

    def __init__(self, data_dir, seq_len=3, transform=None):

        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.transform = transform

        # collect all .tif images and sort by filename (date order)
        self.image_paths = sorted([
            p for p in self.data_dir.glob("*.tif")
        ])
        if len(self.image_paths) < seq_len + 1:
            raise ValueError(f"Not enough frames in {data_dir}")

        # preload metadata: numeric sort if filenames contain dates
        print(f"Loaded {len(self.image_paths)} frames from {data_dir}")

    def __len__(self):
        return len(self.image_paths) - self.seq_len

    def __getitem__(self, idx):
        # sequence of frames [idx : idx+seq_len] → predict next frame
        seq_paths = self.image_paths[idx : idx + self.seq_len]
        target_path = self.image_paths[idx + self.seq_len]

        # load frames into tensor
        frames = [self._load_image(p) for p in seq_paths]
        target = self._load_image(target_path)

        frames = np.stack(frames, axis=0)  # (T, H, W, C)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
        target = np.transpose(target, (2, 0, 1))  # (C, H, W)

        # convert to torch tensor
        frames = torch.from_numpy(frames).float()
        target = torch.from_numpy(target).float()

        if self.transform:
            frames = self.transform(frames)
            target = self.transform(target)

        return frames, target

    def _load_image(self, path):
        img = Image.open(path).convert("RGB").resize((256, 256))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr


def get_dataloader(data_dir, batch_size=4, seq_len=7, shuffle=True, num_workers=0):
    dataset = SatelliteSequenceDataset(data_dir, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    # quick test
    data_root = Path("data_imgs/light_cloud")
    loader = get_dataloader(data_root, batch_size=2, seq_len=7)

    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: x={x.shape}, y={y.shape}")
        if i == 1:
            break
