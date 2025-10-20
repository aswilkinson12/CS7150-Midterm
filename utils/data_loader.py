"""
data_loader.py - HAB Prediction (NOT inpainting)
Predicts frame t using frames t-5 to t-1
"""

import os
import numpy as np
import pandas as pd
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import timedelta
import config as cfg
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


def read_image(path):
    """Read GeoTIFF."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
        with rio.open(path) as src:
            dn = src.read(1).astype(np.float32)
    return dn


def normalize_robust(image, mask):
    """P2-P98 normalization on masked pixels."""
    vals = image[mask]
    if len(vals) < 10:
        return np.zeros_like(image), 0, 1  # ADD p2, p98

    p2, p98 = np.percentile(vals, [2, 98])
    if p98 - p2 < 1e-6:
        p98 = p2 + 1.0

    normed = np.clip((image - p2) / (p98 - p2), 0, 1)
    return normed, p2, p98


class HABPredictionDataset(Dataset):
    """
    Dataset for HAB prediction.
    Input: Past 5 frames (t-5 to t-1)
    Target: Next frame (t)
    """

    def __init__(self, index_df, split='train'):
        self.split = split
        self.index = index_df[index_df['split'] == split].reset_index(drop=True)
        self.index['date'] = pd.to_datetime(self.index['date'])
        self.index = self.index.sort_values('date').reset_index(drop=True)

        self.samples = self._build_samples()
        print(f"[{split}] {len(self.samples)} samples")

    def _build_samples(self):
        """Build prediction samples with temporal sequences."""
        samples = []
        dates_dict = dict(zip(self.index['date'], self.index['path']))

        skipped_water = 0
        skipped_bloom = 0
        skipped_weak = 0
        skipped_gaps = 0

        for idx, row in self.index.iterrows():
            target_date = row['date']
            target_path = row['path']

            full_path = os.path.join(cfg.DATA_DIR, target_path)
            if not os.path.exists(full_path):
                continue

            try:
                dn = read_image(full_path)

                # Water mask
                water_mask = (dn < 254)

                # FILTER 1: Need some water
                if water_mask.sum() < cfg.MIN_WATER_PIXELS:
                    skipped_water += 1
                    continue

                # FILTER 2: Need bloom coverage
                bloom_mask = (dn >= 1) & (dn <= 253)
                bloom_coverage = bloom_mask.sum() / water_mask.sum() if water_mask.sum() > 0 else 0

                if bloom_coverage < cfg.MIN_BLOOM_COVERAGE:
                    skipped_bloom += 1
                    continue

                # FILTER 3: Need sufficient bloom intensity
                bloom_mean = dn[bloom_mask].mean() if bloom_mask.any() else 0
                if bloom_mean < cfg.MIN_BLOOM_INTENSITY:
                    skipped_weak += 1
                    continue

            except Exception as e:
                continue

            # Find past neighbors (t-5 to t-1)
            neighbors = []
            present_bits = []
            has_large_gap = False

            for offset in range(-cfg.LOOKBACK, 0):  # -5, -4, -3, -2, -1
                neighbor_date = target_date + timedelta(days=offset)

                if neighbor_date in dates_dict:
                    neighbors.append(dates_dict[neighbor_date])
                    present_bits.append(1.0)
                else:
                    # Find closest
                    closest_date = None
                    min_gap = float('inf')

                    for available_date in dates_dict.keys():
                        gap = abs((available_date - neighbor_date).days)
                        if gap < min_gap:
                            min_gap = gap
                            closest_date = available_date

                    if closest_date and min_gap <= cfg.MAX_GAP_DAYS:
                        neighbors.append(dates_dict[closest_date])
                        present_bits.append(1.0)
                    else:
                        neighbors.append(None)
                        present_bits.append(0.0)

                        # Critical neighbors missing
                        if abs(offset) <= 1:
                            has_large_gap = True

            # FILTER 4: Skip if critical gaps
            if has_large_gap:
                skipped_gaps += 1
                continue

            # FILTER 5: Need at least 3 neighbors
            if sum(present_bits) < 3:
                skipped_gaps += 1
                continue

            samples.append({
                'target_path': target_path,
                'target_date': target_date,
                'neighbors': neighbors,
                'present_bits': present_bits
            })

        # Print summary
        if self.split == 'train':
            print(f"\n  Filtering summary:")
            print(f"    Skipped (low water): {skipped_water}")
            print(f"    Skipped (low bloom): {skipped_bloom}")
            print(f"    Skipped (weak blooms): {skipped_weak}")
            print(f"    Skipped (temporal gaps): {skipped_gaps}")
            print(f"    Kept: {len(samples)}\n")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load TARGET frame (what we want to predict)
        target_dn = read_image(os.path.join(cfg.DATA_DIR, sample['target_path']))
        water_mask = (target_dn < 254)

        # Load PAST frames (input to model)
        past_frames = []
        for neighbor_path, present in zip(sample['neighbors'], sample['present_bits']):
            if present and neighbor_path:
                try:
                    neighbor_dn = read_image(os.path.join(cfg.DATA_DIR, neighbor_path))
                    past_frames.append(neighbor_dn)
                except:
                    past_frames.append(np.zeros_like(target_dn))
            else:
                past_frames.append(np.zeros_like(target_dn))

        # Normalize
        target_norm, _, _ = normalize_robust(target_dn, water_mask)
        past_norms = [normalize_robust(f, water_mask)[0] for f in past_frames]

        # Extract random patch
        H, W = target_norm.shape
        max_y = H - cfg.PATCH_SIZE
        max_x = W - cfg.PATCH_SIZE

        if max_y <= 0 or max_x <= 0:
            py, px = 0, 0
        else:
            py = np.random.randint(0, max_y) if self.split == 'train' else max_y // 2
            px = np.random.randint(0, max_x) if self.split == 'train' else max_x // 2

        ps = cfg.PATCH_SIZE
        target_patch = target_norm[py:py + ps, px:px + ps]

        # Past patches
        past_patches = [n[py:py + ps, px:px + ps] for n in past_norms]

        # Pad if needed
        if target_patch.shape != (ps, ps):
            target_patch = np.pad(target_patch,
                                  ((0, ps - target_patch.shape[0]), (0, ps - target_patch.shape[1])))
            past_patches = [np.pad(n, ((0, ps - n.shape[0]), (0, ps - n.shape[1])))
                            for n in past_patches]

        # Build input [T=5, C=6, H, W]
        # C = frame (1) + present_bits (5)
        T = len(past_patches)
        x = np.zeros((T, 2, ps, ps), dtype=np.float32)

        for t in range(T):
            x[t, 0] = past_patches[t]  # Frame
            x[t, 1] = sample['present_bits'][t]

        # Create water mask for loss computation
        water_mask_patch = water_mask[py:py + ps, px:px + ps]
        if water_mask_patch.shape != (ps, ps):
            water_mask_patch = np.pad(water_mask_patch,
                                      ((0, ps - water_mask_patch.shape[0]),
                                       (0, ps - water_mask_patch.shape[1])))

        return (
            torch.from_numpy(x).float(),  # [T, C, H, W]
            torch.from_numpy(target_patch).unsqueeze(0).float(),  # [1, H, W]
            torch.from_numpy(water_mask_patch.astype(np.float32)).unsqueeze(0).float()  # [1, H, W]
        )


def create_dataloaders(index_csv, batch_size=8, num_workers=4):
    """Create dataloaders."""
    index = pd.read_csv(index_csv)

    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = HABPredictionDataset(index, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            drop_last=(split == 'train')
        )

    return loaders