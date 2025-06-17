# ScatterLabel Project Overview

## Purpose
ScatterLabel is a POC tool for visualizing machine learning annotations (bounding boxes) in an interactive 2D scatter plot using embeddings. It helps discover patterns in annotation data through dimensionality reduction.

## Tech Stack
- **Backend**: Python 3.11+
- **Dependencies Management**: uv with pyproject.toml
- **Database**: PostgreSQL (using psycopg3)
- **Embedding Service**: JINA AI API
- **Dimensionality Reduction**: t-SNE (scikit-learn), UMAP (umap-learn)
- **Frontend (planned)**: React with TypeScript, regl-scatterplot, Vite
- **Progress Tracking**: tqdm
- **Type Checking**: mypy with type stubs

## Architecture
1. **Data Export**: `export_labelled_datasets.py` exports dataset metadata from PostgreSQL
2. **Embedding Generation**: `generate_embeddings.py` processes bounding box annotations:
   - Converts bounding boxes to text descriptions
   - Sends to JINA AI for embedding generation
   - Applies dimensionality reduction (t-SNE/UMAP)
   - Outputs JSON with embeddings + 2D coordinates
3. **Visualization**: React frontend with regl-scatterplot (planned)

## Environment Variables
- `POSTGRES_URI`: PostgreSQL connection string
- `JINA_API_KEY`: API key for JINA AI embedding service

## Key Files
- `export_labelled_datasets.py`: Exports from machine_learning.labelled_datasets table
- `generate_embeddings.py`: Main script for embedding generation and dimensionality reduction
- `pyproject.toml`: Project configuration and dependencies
- `IMPLEMENTATION_PLAN.md`: Detailed implementation plan
- `CLAUDE.md`: AI assistant instructions