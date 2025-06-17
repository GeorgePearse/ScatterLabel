# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Discuss implementation plans with ChatGPT, grok and gemini via the zen-mcp-server, and ask them to check your work

## Project Overview

ScatterLabel is a POC tool for visualizing machine learning annotations (bounding boxes) in an interactive 2D scatter plot using embeddings. It helps discover patterns in annotation data through dimensionality reduction.

## Development Setup

Always use uv for dependency management:
```bash
uv sync
```

Required environment variables:
- `POSTGRES_URI` - PostgreSQL connection string for accessing `machine_learning.labelled_datasets`
- `JINA_API_KEY` - API key for JINA AI embedding service

## Key Commands

```bash
# Export dataset metadata from PostgreSQL
python export_labelled_datasets.py

# Generate embeddings for CMR dataset
python generate_embeddings.py --dataset cmr --input annotations.csv --output embeddings.json --tsne --umap

# Custom column mapping example
python generate_embeddings.py --input custom.csv --output output.json --x-min-col left --y-min-col top --x-max-col right --y-max-col bottom
```

## Architecture

The pipeline follows these steps:
1. **Data Export**: `export_labelled_datasets.py` connects to PostgreSQL and exports dataset metadata
2. **Embedding Generation**: `generate_embeddings.py` processes bounding box annotations:
   - Converts bounding boxes to text descriptions (size, aspect ratio, location)
   - Sends descriptions to JINA AI for embedding generation
   - Applies dimensionality reduction (t-SNE/UMAP) for 2D visualization
   - Outputs JSON with original annotations + embeddings + 2D coordinates
3. **Visualization**: React frontend with regl-scatterplot (planned implementation)

## Important Implementation Notes

- The system currently supports COCO, YOLO, and custom CSV formats
- Bounding boxes are normalized to [0,1] range before text description
- JINA AI embeddings use the 'jina-embeddings-v3' model with 'retrieval.passage' task
- When implementing frontend, use `regl-scatterplot` for performance with large datasets
- Always add tqdm progress bars for long-running operations
- Use psycopg (psycopg3) not psycopg2 for PostgreSQL connections

## Testing

When testing instance segmentation setups, use these CMR dataset files:
- Train: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
- Val: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
- Images: `/home/georgepearse/data/images/`

Always verify the number of classes in annotation files as model architectures must match.
