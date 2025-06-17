# ScatterLabel

A tool for visualizing machine learning annotations (bounding boxes) in an interactive 2D scatter plot using embeddings.

## Overview

ScatterLabel helps discover patterns in annotation data by:
1. Converting bounding box annotations to text descriptions
2. Generating embeddings using JINA AI
3. Reducing dimensions using t-SNE/UMAP
4. Visualizing in an interactive scatter plot

## Quick Start

### 1. Generate Embeddings

```bash
# Sample 100 diverse crops from dataset
python sample_and_generate_embeddings.py --n-samples 100

# Generate embeddings with t-SNE projection
python generate_embeddings.py --input sampled_annotations.csv --output embeddings_with_ids.json --tsne

# Generate embeddings with image cropping
python generate_embeddings.py --input sampled_annotations.csv --output embeddings_with_ids.json \
  --crop-images --image-dir /path/to/images --crops-dir crops --tsne --save-csv

# Convert to CSV for frontend (if not using --save-csv)
python convert_embeddings_to_csv.py --input embeddings_with_ids.json --output scatterplot_data.csv
```

### 2. Run Frontend

```bash
# Copy data to frontend
cp scatterplot_data.csv frontend/scatter-viewer/public/

# Start the visualization
cd frontend/scatter-viewer
npm install
npm run dev
```

Open http://localhost:5173 to view the interactive scatter plot.

## Features

- **Interactive Visualization**: Pan, zoom, and explore embeddings
- **Class Coloring**: Points colored by object class
- **Hover Details**: View annotation metadata on hover
- **Lasso Selection**: Select clusters of similar annotations
- **Responsive Design**: Works on different screen sizes
- **Image Cropping**: Automatically crop and save bounding box regions
- **CSV Export**: Export embeddings with cropped image paths

## Requirements

- Python 3.11+
- Node.js 16+
- JINA API key (set as `JINA_API_KEY` environment variable)

## Project Structure

```
ScatterLabel/
├── generate_embeddings.py      # Main embedding generation
├── sample_and_generate_embeddings.py  # Data sampling
├── convert_embeddings_to_csv.py    # Format conversion
├── frontend/scatter-viewer/    # React visualization app
└── crops/                      # Cropped images (generated)
```

## Command Line Options

### generate_embeddings.py

```bash
# Basic usage
python generate_embeddings.py --input annotations.csv --output embeddings.json

# With image cropping
python generate_embeddings.py --input annotations.csv --output embeddings.json \
  --crop-images --image-dir /path/to/images --crops-dir crops

# Additional options
--dataset NAME          # Filter by dataset name
--tsne                  # Include t-SNE projection (default: True)
--umap                  # Include UMAP projection
--normalized-coords     # If bbox coordinates are normalized (0-1)
--save-csv             # Save CSV output (default: True)
--batch-size N         # API batch size (default: 100)

# Custom column mapping
--class-col NAME       # Column for class names (default: class_name)
--x-min-col NAME       # Column for x_min (default: x_min)
--y-min-col NAME       # Column for y_min (default: y_min)
--x-max-col NAME       # Column for x_max (default: x_max)
--y-max-col NAME       # Column for y_max (default: y_max)
```

## Image Cropping

When `--crop-images` is specified:
1. Each bounding box is cropped from its source image
2. Crops are saved with encoded filenames: `{image_id}_{x_min}_{y_min}_{x_max}_{y_max}_{class}_{hash}.jpg`
3. The cropped image path is included in the output JSON and CSV
4. Failed crops are logged but don't stop processing