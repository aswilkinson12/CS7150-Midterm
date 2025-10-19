"""
create_index.py - Generate index.csv for CyAN data
"""

import os
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "../data/raw"

# Scan all files
files = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.tif'):
        continue

    try:
        # Parse: L2016153.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_7_2.tif
        # Format: L{YEAR}{DAY_OF_YEAR}.{rest}
        parts = fname.split('.')
        date_part = parts[0]  # "L2016153"

        # Extract year and day-of-year
        year = int(date_part[1:5])      # "2016"
        day_of_year = int(date_part[5:])  # "153"

        # Convert to date
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        # Extract tile from filename (7_2)
        tile = fname.split('_')[-2] + '_' + fname.split('_')[-1].replace('.tif', '')

        files.append({
            'path': fname,
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'tile': tile
        })

    except Exception as e:
        print(f"Skipping {fname}: {e}")
        continue

if len(files) == 0:
    print("No valid files found! Check DATA_DIR path.")
    exit(1)

df = pd.DataFrame(files)
df = df.sort_values('date').reset_index(drop=True)

# Create splits (70/15/15)
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

df['split'] = 'test'
df.loc[:train_end, 'split'] = 'train'
df.loc[train_end:val_end, 'split'] = 'val'

# Print summary
print("\n" + "="*50)
print(f"✓ Processed {len(df)} files")
print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Months: {sorted(df['month'].unique())}")
print(f"\nSplit distribution:")
print(df['split'].value_counts().sort_index())
print("="*50)

# Save
df.to_csv('../data/index.csv', index=False)
print(f"\n✓ Saved to data/index.csv")
print(f"\nFirst few rows:")
print(df.head())