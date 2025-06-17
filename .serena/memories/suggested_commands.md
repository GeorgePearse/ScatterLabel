# Suggested Commands for ScatterLabel Development

## Environment Setup
```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Running the Application
```bash
# Export dataset metadata from PostgreSQL
python export_labelled_datasets.py

# Generate embeddings for CMR dataset
python generate_embeddings.py --dataset cmr --input annotations.csv --output embeddings.json --tsne --umap

# Custom column mapping example
python generate_embeddings.py --input custom.csv --output output.json \
  --x-min-col left --y-min-col top --x-max-col right --y-max-col bottom
```

## Development Commands
```bash
# Type checking with mypy
mypy .

# Run main script
python main.py

# Add and commit to check pre-commit hooks
git add .
git commit -m "Check pre-commit hooks"
```

## Testing with CMR Dataset
```bash
# Generate embeddings for CMR dataset specifically
python generate_embeddings.py --dataset cmr --input labelled_datasets.csv --output cmr_embeddings.json --tsne --umap
```

## Environment Variables Setup
```bash
# Create .env file with:
export POSTGRES_URI="postgresql://user:password@host:port/database"
export JINA_API_KEY="your-jina-api-key"

# Load environment
source .env
```

## System Commands (Linux)
```bash
# Git operations
git status
git add .
git commit -m "message"
git log --oneline

# File operations
ls -la
find . -name "*.py"
grep -r "pattern" .
rg "pattern"  # ripgrep (preferred)

# Process monitoring
ps aux | grep python
htop
```