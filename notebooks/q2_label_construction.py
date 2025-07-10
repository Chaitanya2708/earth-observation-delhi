# q2_label_construction.py

import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rasterio.windows import Window
import random
import os

# ---------- Setup ----------
os.makedirs("outputs", exist_ok=True)
np.random.seed(42)
random.seed(42)

# ---------- ESA WorldCover class mapping ----------
esa_to_label = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow / Ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss / Lichen'
}

# ---------- Load filtered image coordinates ----------
df = pd.read_csv("outputs/q1_filtered_images.csv")
print(f"Total filtered images: {len(df)}")

# ---------- Open the land cover raster ----------
raster_path = "data/land_cover.tif"
raster = rasterio.open(raster_path)
raster_crs = raster.crs
print(f"Loaded raster with CRS: {raster_crs}")

# ---------- Convert image coordinates to raster CRS ----------
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf.to_crs(raster_crs)

# ---------- Label extraction ----------
labeled_data = []

for i, row in gdf.iterrows():
    x, y = row.geometry.x, row.geometry.y
    filename = row['filename']
    
    try:
        # Convert to row, col in raster
        row_col = raster.index(x, y)
        center_row, center_col = row_col

        half_size = 64
        window = Window(center_col - half_size, center_row - half_size, 128, 128)
        patch = raster.read(1, window=window)

        if patch.shape != (128, 128):
            print(f"Skipping {filename}: patch shape {patch.shape}")
            continue

        patch_flat = patch.flatten()
        patch_valid = patch_flat[patch_flat > 0]

        if len(patch_valid) == 0:
            print(f"Skipping {filename}: only no-data values")
            continue

        mode_code = Counter(patch_valid).most_common(1)[0][0]
        mode_label = esa_to_label.get(mode_code, "Unknown")

        labeled_data.append({
            "filename": filename,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "class_code": mode_code,
            "class_name": mode_label
        })

    except Exception as e:
        print(f"Error with {filename}: {e}")

print(f" Labeled {len(labeled_data)} images")

# ---------- Save to CSV ----------
labeled_df = pd.DataFrame(labeled_data)

# Split into train/test
labeled_df["split"] = np.where(np.random.rand(len(labeled_df)) < 0.6, "train", "test")

labeled_df.to_csv("outputs/q2_labeled_dataset.csv", index=False)
print(" Saved labeled dataset: outputs/q2_labeled_dataset.csv")

# ---------- Visualize class distribution ----------
plt.figure(figsize=(10, 5))
class_counts = labeled_df['class_name'].value_counts().sort_values(ascending=False)
class_counts.plot(kind='bar', color='skyblue')
plt.title("Land Cover Class Distribution")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/q2_class_distribution.png")
plt.close()
print(" Saved class distribution plot: outputs/q2_class_distribution.png")
