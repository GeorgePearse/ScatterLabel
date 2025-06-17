#!/usr/bin/env python3
"""
Generate embeddings from bounding box data using JINA AI API for regl-scatterplot visualization.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import json
import time
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class BoundingBox:
    """Represents a bounding box with class information."""
    class_name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_id: Optional[str] = None
    confidence: Optional[float] = None
    annotation_id: Optional[str] = None


class JINAEmbeddingGenerator:
    """Generates embeddings using JINA AI API."""
    
    JINA_API_URL = "https://api.jina.ai/v1/embeddings"
    MAX_BATCH_SIZE = 100  # JINA API batch limit
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: JINA AI API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA API key not provided. Set JINA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _prepare_text_from_bbox(self, bbox: BoundingBox) -> str:
        """
        Convert bounding box information to text for embedding.
        
        Args:
            bbox: BoundingBox instance
            
        Returns:
            Text representation of the bounding box
        """
        # Create a descriptive text representation
        text_parts = [
            f"class: {bbox.class_name}",
            f"position: ({bbox.x_min:.2f}, {bbox.y_min:.2f})",
            f"size: {bbox.x_max - bbox.x_min:.2f}x{bbox.y_max - bbox.y_min:.2f}"
        ]
        
        if bbox.image_id:
            text_parts.append(f"image: {bbox.image_id}")
        
        if bbox.confidence is not None:
            text_parts.append(f"confidence: {bbox.confidence:.3f}")
        
        return " | ".join(text_parts)
    
    def _call_jina_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call JINA API to get embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        payload = {
            "input": texts,
            "model": "jina-embeddings-v2-base-en"  # You can change this to other models
        }
        
        try:
            response = requests.post(
                self.JINA_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"JINA API request failed: {e}")
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Invalid JINA API response format: {e}")
    
    def generate_embeddings(
        self,
        bounding_boxes: List[BoundingBox],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of bounding boxes.
        
        Args:
            bounding_boxes: List of BoundingBox instances
            batch_size: Batch size for API calls (default: MAX_BATCH_SIZE)
            
        Returns:
            NumPy array of embeddings (shape: [n_boxes, embedding_dim])
        """
        if not bounding_boxes:
            return np.array([])
        
        batch_size = min(batch_size or self.MAX_BATCH_SIZE, self.MAX_BATCH_SIZE)
        
        # Convert bounding boxes to text
        texts = [self._prepare_text_from_bbox(bbox) for bbox in bounding_boxes]
        
        # Process in batches with progress bar
        all_embeddings = []
        
        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        batch_embeddings = self._call_jina_api(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                        break
                    except RuntimeError as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                pbar.update(len(batch_texts))
                
                # Rate limiting
                if i + batch_size < len(texts):
                    time.sleep(0.1)  # Small delay between batches
        
        return np.array(all_embeddings)


def load_bounding_boxes_from_dataframe(
    df: pd.DataFrame,
    class_col: str = "class_name",
    bbox_cols: Optional[Dict[str, str]] = None,
    image_col: Optional[str] = None,
    confidence_col: Optional[str] = None,
    annotation_col: Optional[str] = None
) -> List[BoundingBox]:
    """
    Load bounding boxes from a pandas DataFrame.
    
    Args:
        df: DataFrame containing bounding box data
        class_col: Column name for class names
        bbox_cols: Dictionary mapping bbox attributes to column names
                  Default: {"x_min": "x_min", "y_min": "y_min", 
                           "x_max": "x_max", "y_max": "y_max"}
        image_col: Optional column name for image IDs
        confidence_col: Optional column name for confidence scores
        annotation_col: Optional column name for annotation IDs
        
    Returns:
        List of BoundingBox instances
    """
    if bbox_cols is None:
        bbox_cols = {
            "x_min": "x_min",
            "y_min": "y_min",
            "x_max": "x_max",
            "y_max": "y_max"
        }
    
    bounding_boxes = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading bounding boxes"):
        bbox = BoundingBox(
            class_name=row[class_col],
            x_min=float(row[bbox_cols["x_min"]]),
            y_min=float(row[bbox_cols["y_min"]]),
            x_max=float(row[bbox_cols["x_max"]]),
            y_max=float(row[bbox_cols["y_max"]]),
            image_id=row.get(image_col) if image_col else None,
            confidence=float(row[confidence_col]) if confidence_col and confidence_col in row else None,
            annotation_id=str(row[annotation_col]) if annotation_col and annotation_col in row else None
        )
        bounding_boxes.append(bbox)
    
    return bounding_boxes


def save_embeddings_for_scatterplot(
    embeddings: np.ndarray,
    bounding_boxes: List[BoundingBox],
    output_path: str,
    include_tsne: bool = True,
    include_umap: bool = False
) -> None:
    """
    Save embeddings in a format suitable for regl-scatterplot.
    
    Args:
        embeddings: NumPy array of embeddings
        bounding_boxes: List of corresponding BoundingBox instances
        output_path: Path to save the output JSON file
        include_tsne: Whether to include t-SNE projection (requires scikit-learn)
        include_umap: Whether to include UMAP projection (requires umap-learn)
    """
    data = {
        "embeddings": embeddings.tolist(),
        "metadata": []
    }
    
    # Add metadata for each point
    for i, bbox in enumerate(bounding_boxes):
        metadata = {
            "index": i,
            "class_name": bbox.class_name,
            "bbox": {
                "x_min": bbox.x_min,
                "y_min": bbox.y_min,
                "x_max": bbox.x_max,
                "y_max": bbox.y_max
            }
        }
        
        if bbox.image_id:
            metadata["image_id"] = bbox.image_id
        
        if bbox.confidence is not None:
            metadata["confidence"] = bbox.confidence
        
        if bbox.annotation_id is not None:
            metadata["annotation_id"] = bbox.annotation_id
        
        data["metadata"].append(metadata)
    
    # Add 2D projections if requested
    if include_tsne or include_umap:
        print("Computing 2D projections...")
        
        if include_tsne:
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                tsne_coords = tsne.fit_transform(embeddings)
                data["tsne"] = tsne_coords.tolist()
            except ImportError:
                print("Warning: scikit-learn not installed, skipping t-SNE projection")
        
        if include_umap:
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                umap_coords = reducer.fit_transform(embeddings)
                data["umap"] = umap_coords.tolist()
            except ImportError:
                print("Warning: umap-learn not installed, skipping UMAP projection")
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved embeddings to {output_path}")


def filter_annotations_by_dataset(annotations_df: pd.DataFrame, dataset_name: str, datasets_csv_path: str = "labelled_datasets.csv") -> pd.DataFrame:
    """
    Filter annotations to only include those from a specific dataset.
    
    Args:
        annotations_df: DataFrame with annotations
        dataset_name: Name of the dataset to filter for
        datasets_csv_path: Path to the labelled_datasets.csv file
        
    Returns:
        Filtered DataFrame
    """
    # Load the labelled datasets
    datasets_df = pd.read_csv(datasets_csv_path)
    
    # Get image IDs for the specified dataset
    dataset_images = datasets_df[datasets_df['dataset_name'] == dataset_name]['image_id'].tolist()
    
    if not dataset_images:
        raise ValueError(f"No images found for dataset '{dataset_name}'")
    
    # Filter annotations to only include those from the specified dataset
    filtered_df = annotations_df[annotations_df['image_id'].isin(dataset_images)]
    
    print(f"Filtered {len(filtered_df)} annotations from dataset '{dataset_name}' (out of {len(annotations_df)} total)")
    
    return filtered_df


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Generate embeddings for annotation visualization")
    parser.add_argument("--input", "-i", required=True, help="Path to input annotations CSV file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file with embeddings")
    parser.add_argument("--dataset", "-d", help="Filter annotations by dataset name (from labelled_datasets.csv)")
    parser.add_argument("--datasets-csv", default="labelled_datasets.csv", help="Path to labelled_datasets.csv")
    parser.add_argument("--tsne", action="store_true", default=True, help="Include t-SNE projection (default: True)")
    parser.add_argument("--no-tsne", dest="tsne", action="store_false", help="Exclude t-SNE projection")
    parser.add_argument("--umap", action="store_true", default=False, help="Include UMAP projection (default: False)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for API calls (default: 100)")
    
    # Columns configuration
    parser.add_argument("--class-col", default="class_name", help="Column name for class names")
    parser.add_argument("--image-col", default="image_id", help="Column name for image IDs")
    parser.add_argument("--confidence-col", default="confidence", help="Column name for confidence scores")
    parser.add_argument("--x-min-col", default="x_min", help="Column name for x_min")
    parser.add_argument("--y-min-col", default="y_min", help="Column name for y_min")
    parser.add_argument("--x-max-col", default="x_max", help="Column name for x_max")
    parser.add_argument("--y-max-col", default="y_max", help="Column name for y_max")
    parser.add_argument("--annotation-col", default="annotation_id", help="Column name for annotation IDs")
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.input}")
    annotations_df = pd.read_csv(args.input)
    print(f"Loaded {len(annotations_df)} annotations")
    
    # Filter by dataset if specified
    if args.dataset:
        annotations_df = filter_annotations_by_dataset(
            annotations_df, 
            args.dataset, 
            args.datasets_csv
        )
    
    # Set up bbox column mapping
    bbox_cols = {
        "x_min": args.x_min_col,
        "y_min": args.y_min_col,
        "x_max": args.x_max_col,
        "y_max": args.y_max_col
    }
    
    # Load bounding boxes
    bboxes = load_bounding_boxes_from_dataframe(
        annotations_df,
        class_col=args.class_col,
        bbox_cols=bbox_cols,
        image_col=args.image_col,
        confidence_col=args.confidence_col,
        annotation_col=args.annotation_col
    )
    
    # Initialize generator
    generator = JINAEmbeddingGenerator()
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(bboxes, batch_size=args.batch_size)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save for visualization
    save_embeddings_for_scatterplot(
        embeddings,
        bboxes,
        args.output,
        include_tsne=args.tsne,
        include_umap=args.umap
    )


if __name__ == "__main__":
    main()