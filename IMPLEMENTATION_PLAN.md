# ScatterLabel POC - Regl-Scatterplot for Annotation Visualization

## Overview
This document outlines the implementation plan for a proof-of-concept that uses regl-scatterplot to visualize annotated images based on their embeddings. The POC consists of:
1. A Python script that generates embeddings and UMAP projections for annotations
2. A React frontend that reads from a local CSV and displays an interactive scatterplot with ImageKit-powered image gallery

**Current Focus**: We are testing this implementation specifically with the CMR dataset first to validate the full system before expanding to other datasets.

## Architecture Overview

### Component Structure
```
scatterlabel-poc/
├── python/
│   ├── generate_embeddings.py    # Generates embeddings and adds X,Y coordinates to CSV
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ScatterplotViewer/
│   │   │   │   ├── index.tsx
│   │   │   │   └── ScatterplotCanvas.tsx
│   │   │   ├── ImageGallery/
│   │   │   │   ├── index.tsx
│   │   │   │   └── ImageCard.tsx
│   │   │   └── CSVLoader/
│   │   │       └── index.tsx
│   │   ├── hooks/
│   │   │   └── useCSVData.ts
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
└── data/
    └── annotations_with_embeddings.csv  # CSV with X, Y columns added
```

## Data Flow Architecture

### 1. Data Processing Pipeline (Python)
```
Annotation CSV → Load Annotations → Generate Embeddings → UMAP Projection → Save CSV with X,Y
```

### 2. Frontend Data Flow
```
CSV File → Parse in Browser → Regl-Scatterplot → User Selection → Filter Images → Display Gallery
```

### 3. CSV Structure
```csv
image_id,bbox_x,bbox_y,bbox_w,bbox_h,class_name,confidence,embedding_x,embedding_y
img_001.jpg,100,200,50,60,person,0.95,0.123,0.456
img_002.jpg,300,400,80,90,car,0.87,-0.234,0.789
...
```

## Implementation Steps

### Step 1: Python Embedding Generation Script
1. **Script Features**
   ```python
   # generate_embeddings.py
   - Load annotations from CSV
   - Extract image crops based on bounding boxes
   - Generate embeddings using JINA AI or other model
   - Apply UMAP for 2D projection
   - Save augmented CSV with X, Y coordinates
   ```

2. **Dependencies**
   ```txt
   pandas
   numpy
   umap-learn
   pillow
   jina
   tqdm
   ```

### Step 2: React Frontend
1. **Regl-Scatterplot Integration**
   ```typescript
   interface DataPoint {
     x: number;
     y: number;
     imageId: string;
     bbox: [number, number, number, number];
     className: string;
     confidence: number;
   }
   ```

2. **Key Components**
   - CSV file picker/loader
   - Scatterplot with selection tools
   - Selected images gallery
   - Class filter checkboxes
   - Confidence threshold slider

3. **ImageKit Integration**
   - Dynamic crop generation from bounding boxes
   - On-demand image loading
   - Responsive thumbnails

## Technical Stack

### Python Script
- **pandas** for CSV manipulation
- **numpy** for numerical operations
- **umap-learn** for dimensionality reduction
- **Pillow** for image processing
- **JINA AI SDK** or alternative embedding model
- **tqdm** for progress tracking

### Frontend
- **React** with TypeScript
- **regl-scatterplot** for WebGL visualization
- **Papa Parse** for CSV parsing
- **Tailwind CSS** for styling
- **Vite** for build tooling
- **ImageKit React SDK** for image display

## Quick Start Guide

### 1. Generate Embeddings
```bash
cd python
pip install -r requirements.txt
# For testing with CMR dataset specifically:
python generate_embeddings.py --dataset cmr --input annotations.csv --output annotations_with_embeddings.csv
# The script supports filtering by dataset name from labelled_datasets.csv
```

### 2. Run Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Load CSV in Browser
- Click "Load CSV" button
- Select the generated CSV file
- Interact with the scatterplot
- View selected images in the gallery

## Example Code Snippets

### Python Embedding Generation
```python
import pandas as pd
import numpy as np
from umap import UMAP
from PIL import Image
from tqdm import tqdm

def generate_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    embeddings = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Extract crop based on bbox
        crop = extract_crop(row['image_id'], row[['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']])
        # Generate embedding
        embedding = model.encode(crop)
        embeddings.append(embedding)
    
    # Apply UMAP
    reducer = UMAP(n_components=2)
    coords = reducer.fit_transform(embeddings)
    
    df['embedding_x'] = coords[:, 0]
    df['embedding_y'] = coords[:, 1]
    
    return df
```

### React Scatterplot Component
```typescript
import Scatterplot from 'regl-scatterplot';

const ScatterplotViewer: React.FC = () => {
  const [selection, setSelection] = useState<number[]>([]);
  
  useEffect(() => {
    const scatterplot = new Scatterplot({
      canvas,
      lassoMinDelay: 10,
      lassoMinDist: 4,
      showReticle: true,
      reticleColor: [1, 1, 0.878, 0.33]
    });
    
    scatterplot.subscribe('select', ({ points }) => {
      setSelection(points);
    });
    
    return () => scatterplot.destroy();
  }, []);
  
  return <canvas ref={canvasRef} />;
};
```

## POC Limitations & Future Work

This POC demonstrates the core concept. Future enhancements could include:
- Server-side embedding generation API
- Real-time updates as new annotations arrive
- Multiple embedding models comparison
- Export functionality for selected subsets
- Integration with annotation tools