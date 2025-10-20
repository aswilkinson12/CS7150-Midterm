import os
import numpy as np
import pandas as pd
import rasterio as rio
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import timedelta
from scipy.ndimage import distance_transform_edt, gaussian_filter
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


def generate_synthetic_mask(shape, water_mask, coverage_range=(0.10, 0.40)):
    """Generate synthetic cloud mask."""
    H, W = shape

    if not water_mask.any():
        return np.zeros((H, W), dtype=np.float32)

    # Target coverage
    target_cov = np.random.uniform(*coverage_range)

    # Create random blobs
    mask = np.zeros((H, W), dtype=np.float32)
    num_blobs = np.random.randint(3, 8)

    for _ in range(num_blobs):
        # Random center in water
        water_coords = np.argwhere(water_mask)
        if len(water_coords) == 0:
            break
        cy, cx = water_coords[np.random.randint(len(water_coords))]

        # Random size
        r_y = np.random.randint(30, 100)
        r_x = np.random.randint(30, 100)

        # Create blob
        Y, X = np.ogrid[:H, :W]
        dist = ((Y - cy) / r_y)**2 + ((X - cx) / r_x)**2
        blob = (dist < 1.0).astype(np.float32)
        mask = np.maximum(mask, blob)

    # Smooth
    mask = gaussian_filter(mask, sigma=8)

    # Threshold to match coverage
    if mask.max() > 0:
        water_pixels = water_mask.sum()
        target_pixels = int(water_pixels * target_cov)
        sorted_vals = np.sort(mask[water_mask].flatten())
        if len(sorted_vals) > target_pixels:
            threshold = sorted_vals[-target_pixels]
            mask = (mask > threshold).astype(np.float32)
        else:
            mask = (mask > 0.1).astype(np.float32)

    # Restrict to water
    mask = mask * water_mask.astype(np.float32)

    return mask


def normalize_robust(image, mask):
    """P2-P98 normalization on masked pixels."""
    vals = image[mask]
    if len(vals) < 10:
        return np.zeros_like(image)

    p2, p98 = np.percentile(vals, [2, 98])
    if p98 - p2 < 1e-6:
        p98 = p2 + 1.0

    normed = np.clip((image - p2) / (p98 - p2), 0, 1)
    return normed


class HABInpaintDataset(Dataset):
    """Dataset with cloud filtering and synthetic masks."""

    def __init__(self, index_df, split='train'):
        self.split = split
        self.index = index_df[index_df['split'] == split].reset_index(drop=True)
        self.index['date'] = pd.to_datetime(self.index['date'])
        self.index = self.index.sort_values('date').reset_index(drop=True)

        self.samples = self._build_samples()
        print(f"[{split}] {len(self.samples)} samples")

    def _build_samples(self):
        """Build sample list - filter cloudy targets, keep cloudy neighbors."""
        samples = []
        dates_dict = dict(zip(self.index['date'], self.index['path']))

        skipped_cloudy = 0
        skipped_water = 0
        skipped_gaps = 0

        for idx, row in self.index.iterrows():
            target_date = row['date']
            target_path = row['path']

            full_path = os.path.join(cfg.DATA_DIR, target_path)
            if not os.path.exists(full_path):
                continue

            try:
                dn = read_image(full_path)

                # Water mask (DN < 254)
                water_mask = (dn < 254)
                cloud_mask = (dn == 255)

                # FILTER 1: Skip if target is too cloudy (>95%)
                cloud_coverage = cloud_mask.sum() / dn.size
                if cloud_coverage > 0.95:
                    skipped_cloudy += 1
                    continue

                # FILTER 2: Skip if not enough visible water
                visible_water = water_mask & ~cloud_mask
                if visible_water.sum() < cfg.MIN_WATER_PIXELS:
                    skipped_water += 1
                    continue

            except Exception as e:
                continue

            # Find temporal neighbors (can be cloudy - that's OK!)
            neighbors = []
            present_bits = []
            has_large_gap = False

            for offset in range(-cfg.LOOKBACK, 0):
                neighbor_date = target_date + timedelta(days=offset)

                # Check if neighbor exists
                if neighbor_date in dates_dict:
                    neighbors.append(dates_dict[neighbor_date])
                    present_bits.append(1.0)
                else:
                    # Find closest available date
                    closest_date = None
                    min_gap = float('inf')

                    for available_date in dates_dict.keys():
                        gap = abs((available_date - neighbor_date).days)
                        if gap < min_gap:
                            min_gap = gap
                            closest_date = available_date

                    # Check if gap is acceptable
                    if closest_date and min_gap <= cfg.MAX_GAP_DAYS:
                        neighbors.append(dates_dict[closest_date])
                        present_bits.append(1.0)
                    else:
                        # Gap too large
                        neighbors.append(None)
                        present_bits.append(0.0)

                        # If critical neighbor missing
                        if abs(offset) <= 1:
                            has_large_gap = True

            # FILTER 3: Skip if temporal gaps too large
            if has_large_gap:
                skipped_gaps += 1
                continue

            # FILTER 4: Need at least 3 neighbors
            if sum(present_bits) < 3:
                skipped_gaps += 1
                continue

            samples.append({
                'target_path': target_path,
                'neighbors': neighbors,
                'present_bits': present_bits
            })

        # Print filtering summary
        if self.split == 'train':
            print(f"\n  Filtering summary:")
            print(f"    Skipped (>95% cloudy): {skipped_cloudy}")
            print(f"    Skipped (low water): {skipped_water}")
            print(f"    Skipped (temporal gaps): {skipped_gaps}")
            print(f"    Kept: {len(samples)}\n")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load target
        target_dn = read_image(os.path.join(cfg.DATA_DIR, sample['target_path']))

        # Water mask (DN < 254)
        water_mask = (target_dn < 254)

        # Generate synthetic cloud mask (only on visible water)
        synth_mask = generate_synthetic_mask(
            target_dn.shape, water_mask, cfg.SYNTH_MASK_COVERAGE
        )

        # Load neighbors (these CAN be cloudy - model learns to handle it)
        neighbor_frames = []
        for neighbor_path, present in zip(sample['neighbors'], sample['present_bits']):
            if present and neighbor_path:
                try:
                    neighbor_dn = read_image(os.path.join(cfg.DATA_DIR, neighbor_path))
                    neighbor_frames.append(neighbor_dn)
                except:
                    neighbor_frames.append(np.zeros_like(target_dn))
            else:
                neighbor_frames.append(np.zeros_like(target_dn))

        # Normalize
        target_norm = normalize_robust(target_dn, water_mask)
        neighbor_norms = [normalize_robust(f, water_mask) for f in neighbor_frames]

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
        target_patch = target_norm[py:py+ps, px:px+ps]
        mask_patch = synth_mask[py:py+ps, px:px+ps]

        # Neighbor patches
        neighbor_patches = [n[py:py+ps, px:px+ps] for n in neighbor_norms]

        # Pad if needed
        if target_patch.shape != (ps, ps):
            target_patch = np.pad(target_patch, ((0, ps-target_patch.shape[0]), (0, ps-target_patch.shape[1])))
            mask_patch = np.pad(mask_patch, ((0, ps-mask_patch.shape[0]), (0, ps-mask_patch.shape[1])))
            neighbor_patches = [np.pad(n, ((0, ps-n.shape[0]), (0, ps-n.shape[1]))) for n in neighbor_patches]

        # Build input [T, C, H, W]
        T = len(neighbor_patches)
        x = np.zeros((5, 7, ps, ps), dtype=np.float32)

        for t in range(T):
            x[t, 0] = neighbor_patches[t]
            x[t, 1] = mask_patch
            for pb_idx, pb_val in enumerate(sample['present_bits']):
                x[t, 2 + pb_idx] = pb_val

        # Distance weights
        if mask_patch.sum() > 0:
            dist = distance_transform_edt(mask_patch > 0.5)
            dist = dist / (dist.max() + 1e-8)
        else:
            dist = np.zeros_like(mask_patch)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(target_patch).unsqueeze(0).float(),
            torch.from_numpy(mask_patch).unsqueeze(0).float(),
            torch.from_numpy(dist).unsqueeze(0).float()
        )


def create_dataloaders(index_csv, batch_size=8, num_workers=4):
    """Create dataloaders."""
    index = pd.read_csv(index_csv)

    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = HABInpaintDataset(index, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            drop_last=(split == 'train')
        )

    return loaders