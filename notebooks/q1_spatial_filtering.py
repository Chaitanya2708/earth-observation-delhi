# q1_spatial_filtering.py

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import geemap
import time

start_total = time.time()

# ---------- STEP 1: Load and project Delhi-NCR shapefile ----------
print(" Loading Delhi-NCR shapefile...")
ncr = gpd.read_file("data/delhi_ncr_region.geojson")
print(f"Original CRS: {ncr.crs}")

# Reproject to EPSG:32644 for metric distances
ncr = ncr.to_crs("EPSG:32644")
print(f"Projected CRS: {ncr.crs}")

# ---------- STEP 2: Create 60×60 km grid ----------
print(" Creating 60×60 km grid...")
grid_size = 60000  # in meters
minx, miny, maxx, maxy = ncr.total_bounds
grid_cells = [box(x, y, x + grid_size, y + grid_size)
              for x in range(int(minx), int(maxx), grid_size)
              for y in range(int(miny), int(maxy), grid_size)]
grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:32644")
print(f"Generated {len(grid)} grid cells.")

# ---------- STEP 3: Save static matplotlib plot ----------
os.makedirs("outputs", exist_ok=True)
print(" Saving static grid plot...")
fig, ax = plt.subplots(figsize=(10, 10))
ncr.plot(ax=ax, facecolor='none', edgecolor='blue', label='Delhi NCR Boundary')
grid.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5, label='Grid')
plt.legend()
plt.title("Delhi-NCR with 60×60 km Grid")
plt.savefig("outputs/q1_matplotlib_plot.png")
plt.close()

# ---------- STEP 4: Load RGB image coordinates ----------
print(" Parsing RGB image filenames...")
image_dir = "data/rgb"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

records = []
for fname in image_files:
    try:
        lat, lon = map(float, fname.replace(".png", "").split("_"))
        records.append((fname, lat, lon))
    except:
        print(f" Skipping malformed filename: {fname}")

image_df = pd.DataFrame(records, columns=["filename", "latitude", "longitude"])
image_gdf = gpd.GeoDataFrame(image_df,
    geometry=gpd.points_from_xy(image_df.longitude, image_df.latitude),
    crs="EPSG:4326")
image_gdf_proj = image_gdf.to_crs("EPSG:32644")
print(f"Total images: {len(image_gdf_proj)}")

# ---------- STEP 5: Fast spatial filtering ----------
print(" Filtering images inside grid cells...")
grid_sindex = grid.sindex
filtered_list = []

start_filter = time.time()
for idx, img in image_gdf_proj.iterrows():
    matches_idx = list(grid_sindex.intersection(img.geometry.bounds))
    for _, cell in grid.iloc[matches_idx].iterrows():
        if cell.geometry.contains(img.geometry):
            filtered_list.append(img)
            break
end_filter = time.time()

filtered = gpd.GeoDataFrame(filtered_list, crs=image_gdf_proj.crs)
print(f" Filtered {len(filtered)} / {len(image_gdf_proj)} images in {end_filter - start_filter:.2f}s")
filtered[['filename', 'latitude', 'longitude']].to_csv("outputs/q1_filtered_images.csv", index=False)

# ---------- STEP 6: Create geemap interactive satellite map ----------
print(" Creating interactive geemap...")
grid_wgs = grid.to_crs("EPSG:4326")
centers = grid.geometry.centroid.to_crs("EPSG:4326")
centers_gdf = gpd.GeoDataFrame(geometry=centers, crs="EPSG:4326")

ncr_wgs = ncr.to_crs("EPSG:4326")
filtered_wgs = filtered.to_crs("EPSG:4326")

m = geemap.Map(center=[28.6, 77.2], zoom=8, ee_initialize=False)
m.add_basemap("SATELLITE")

# Add layers
m.add_gdf(grid_wgs, layer_name="Grid (60x60 km)", style={"color": "red"})
m.add_gdf(centers_gdf, layer_name="Grid Centers", style={"color": "yellow", "radius": 4})
m.add_gdf(ncr_wgs, layer_name="Delhi NCR Boundary", style={"color": "blue"})
m.add_gdf(filtered_wgs, layer_name="Image Patch Centers", style={"color": "lime", "radius": 3})

# Save to HTML
m.to_html("outputs/q1_geemap_satellite.html")
print(" Saved interactive map: outputs/q1_geemap_satellite.html")

print(f"\n✅ Q1 Complete. Total time: {time.time() - start_total:.2f} seconds.")
