# Delhi Airshed Land Use Classification (Scenario-1)

This project is part of the **Sustainability Lab** at **IIT Gandhinagar**. It presents an AI-based audit of the **Delhi Airshed** using Sentinel-2 satellite imagery and ESA WorldCover 2021 data to classify land use and identify pollution-linked spatial patterns.

---

## Folder Structure
earth-observation-delhi/
├── data/ # Raw data (shapefiles, RGB patches, raster)
│ ├── delhi_ncr_region.geojson
│ ├── land_cover.tif
│ └── rgb/ 
├── notebooks/
│ ├── q1_spatial_filtering.py
│ ├── q2_label_construction.py
│ └── q3_model_training.py
├── outputs/
│ ├── q1_filtered_images.csv
│ ├── q1_matplotlib_plot.png
│ ├── q1_geemap_satellite.html
│ ├── q2_labeled_dataset.csv
│ ├── q2_class_distribution.png
│ ├── q3_confusion_matrix.png
│ ├── q3_model_results.csv
│ ├── q3_correct_predictions.png
│ └── q3_incorrect_predictions.png
└── README.md


---

##  Project Workflow

###  Q1: Spatial Reasoning & Filtering

- Loaded Delhi-NCR shapefile and reprojected to EPSG:32644
- Overlaid a 60×60 km grid on the map
- Created interactive satellite view using `geemap`
- Parsed RGB image filenames to extract lat/lon
- Filtered images falling within the grid
- 📎 Output:
  - `q1_filtered_images.csv`
  - `q1_geemap_satellite.html`
  - `q1_matplotlib_plot.png`

---

###  Q2: Label Construction

- Loaded `land_cover.tif` (ESA WorldCover 2021, 10m resolution)
- Extracted 128×128 land cover patches centered on image coordinates
- Assigned land use labels using majority class in the patch
- Handled no-data edge cases
- Created 60/40 train-test split
- Visualized class distribution
- 📎 Output:
  - `q2_labeled_dataset.csv`
  - `q2_class_distribution.png`

---

###  Q3: Model Training & Evaluation

- Used pretrained **ResNet18** model from `torchvision`
- Trained on RGB images to predict land cover classes
- Evaluated using:
  - Custom macro F1 score (scikit-learn)
  - TorchMetrics F1 score
  - Confusion matrix
- Saved 5 correct and 5 incorrect predictions
- 📎 Output:
  - `q3_model_results.csv`
  - `q3_confusion_matrix.png`
  - `q3_correct_predictions.png`
  - `q3_incorrect_predictions.png`

---

##  Results

| Metric                  | Value  |
|-------------------------|--------|
| Custom Macro F1 Score   | ~0.54  |
| TorchMetrics Macro F1   | ~0.54  |
| Epochs Trained          | 5      |
| Image Size              | 128×128 |
| Total Labeled Samples   | ~9216 (varies by filtering) |

---

##  Requirements

Install dependencies (Python 3.8+ recommended):

```bash
pip install geopandas rasterio scikit-learn torch torchvision matplotlib seaborn pandas geemap

---
##  Notes
CRS used for spatial operations: EPSG:32644

ESA WorldCover 2021 was used as the ground truth label source

Label assignment uses the mode value from land cover patches



