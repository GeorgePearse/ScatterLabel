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

# Convert to CSV for frontend
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
└── frontend/scatter-viewer/    # React visualization app
```