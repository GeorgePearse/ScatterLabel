# ScatterLabel Viewer

Interactive visualization of machine learning annotation embeddings using regl-scatterplot.

## Features

- Interactive 2D scatter plot of annotation embeddings
- Color-coded by object class
- Hover tooltips showing annotation details
- Lasso selection for exploring clusters
- Responsive design

## Setup

1. Install dependencies:
```bash
npm install
```

2. Ensure `scatterplot_data.csv` is in the `public` folder

3. Start the development server:
```bash
npm run dev
```

4. Open http://localhost:5173 in your browser

## Data Format

The CSV file should contain:
- `tsne_x`, `tsne_y`: 2D coordinates for visualization
- `class_name`: Object class label
- `annotation_id`: Unique identifier
- `x_min`, `y_min`, `x_max`, `y_max`: Bounding box coordinates
- `confidence`: Detection confidence score

## Interactions

- **Hover**: View annotation details
- **Click and drag**: Lasso select multiple points
- **Scroll**: Zoom in/out
- **Drag**: Pan around the plot