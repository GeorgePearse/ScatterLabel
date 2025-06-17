# CMR Frontend Update

## Overview
The frontend has been updated to display CMR (presumably "CMR dataset") cropped images instead of full images.

## Changes Made

### 1. Data Files
- Generated `cmr_scatterplot_data.csv` (1000 records from sample dataset)
- Generated `cmr_scatterplot_data_limited.csv` (2000 records from limited dataset)
- These files include the `cropped_image_path` field pointing to the crop files

### 2. Frontend Components

#### ScatterPlot.jsx
- Updated to load `cmr_scatterplot_data_limited.csv` instead of the generic `scatterplot_data.csv`
- No other changes needed as it passes the full data objects to ImageGrid

#### ImageGrid.jsx
- Removed image grouping logic (since each crop is a separate annotation)
- Updated to display individual cropped images instead of full images with overlays
- Each crop shows:
  - The cropped image
  - Class name
  - Source image ID
  - Bounding box coordinates
- Grid layout adjusted to show smaller tiles (200px min width vs 300px)

#### App.jsx
- Removed unused `imageBasePath` prop from ImageGrid component

### 3. Image Access
- Created symbolic link: `frontend/scatter-viewer/public/cmr_crops` → `/home/georgepearse/ScatterLabel/cmr_crops`
- This allows the frontend to serve the crop images directly

## Usage
1. Start the frontend server as usual
2. The scatter plot will now load CMR data
3. Selecting points will display the corresponding cropped images
4. Each crop is shown individually with its metadata

## Dataset Info
- Sample dataset: 1,000 annotations
- Limited dataset: 2,000 annotations (currently active)
- Full dataset: 48,325 crop images available in `/cmr_crops`

To switch datasets, modify the CSV file path in ScatterPlot.jsx.