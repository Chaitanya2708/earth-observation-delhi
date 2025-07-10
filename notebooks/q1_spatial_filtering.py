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

# Reproject to EPSG:32644 for distance in meters
ncr = ncr.to_crs("EPSG:32644")
print(f"Projected CRS: {ncr.crs}")

# ---------- STEP 2: Create 60x60 km grid ----------
print(" Creating 60×60 km grid...")
grid_size = 60000  # in meters
minx, miny, maxx, maxy = ncr.total_bounds
grid_cells = []

for x in range(int(minx), int(maxx), grid_size):
    for y in range(int(miny), int(maxy), grid_size):
        grid_cells.append(box(x, y, x + grid_size, y + grid_size))

grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:32644")
print(f"Generated {len(grid)} grid cells.")

# ---------- STEP 3: Plot grid over shapefile ----------
print(" Saving static grid plot...")
os.makedirs("outputs", exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 10))
ncr.plot(ax=ax, facecolor='none', edgecolor='blue')
grid.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5)
plt.title("Delhi-NCR with 60×60 km Grid")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.grid(True)
plt.savefig("outputs/q1_matplotlib_plot.png")
plt.close()
print("Plot saved as outputs/q1_matplotlib_plot.png")

# ---------- STEP 4: Interactive satellite map using geemap (no GEE login) ----------
print(" Generating satellite basemap (no Earth Engine)...")
grid_wgs = grid.to_crs("EPSG:4326")
centers = grid.geometry.centroid
centers_gdf = gpd.GeoDataFrame(geometry=centers, crs=grid.crs).to_crs("EPSG:4326")

m = geemap.Map(center=[28.6, 77.2], zoom=8, ee_initialize=False)  # disables Earth Engine
m.add_basemap("SATELLITE")
m.add_gdf(grid_wgs, layer_name="Grid")
m.add_gdf(centers_gdf, layer_name="Grid Centers", style={"color": "yellow", "radius": 5})
m.to_html("outputs/q1_geemap_satellite.html")
print("Interactive map saved as outputs/q1_geemap_satellite.html")

# ---------- STEP 5: Load image filenames and extract coordinates ----------
print(" Parsing image filenames from data/rgb/ ...")
image_dir = "data/rgb"
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

records = []
for filename in image_files:
    try:
        lat_str, lon_str = filename.replace(".png", "").split("_")
        lat, lon = float(lat_str), float(lon_str)
        records.append((filename, lat, lon))
    except ValueError:
        print(f" Skipping malformed filename: {filename}")

image_df = pd.DataFrame(records, columns=["filename", "latitude", "longitude"])
geometry = [Point(lon, lat) for lat, lon in zip(image_df.latitude, image_df.longitude)]
image_gdf = gpd.GeoDataFrame(image_df, geometry=geometry, crs="EPSG:4326")
image_gdf_proj = image_gdf.to_crs("EPSG:32644")

print(f"Total images: {len(image_gdf_proj)}")

# ---------- STEP 6: Fast spatial filtering using bounding boxes ----------
print(" Filtering images that fall within the grid (bounding box method)...")
grid_sindex = grid.sindex

filtered_list = []
start_filter = time.time()

for idx, img in image_gdf_proj.iterrows():
    possible_matches_idx = list(grid_sindex.intersection(img.geometry.bounds))
    possible_matches = grid.iloc[possible_matches_idx]
    for _, cell in possible_matches.iterrows():
        if cell.geometry.contains(img.geometry):
            filtered_list.append(img)
            break

filtered = gpd.GeoDataFrame(filtered_list, crs=image_gdf_proj.crs)
end_filter = time.time()

print(f" Filtered {len(filtered)} images (out of {len(image_gdf_proj)}) in {end_filter - start_filter:.2f} seconds.")

# ---------- STEP 7: Save filtered list ----------
filtered[['filename', 'latitude', 'longitude']].to_csv("outputs/q1_filtered_images.csv", index=False)
print(" Saved filtered image list to outputs/q1_filtered_images.csv")

print(f"\n Q1 Complete. Total time: {time.time() - start_total:.2f} seconds.")
