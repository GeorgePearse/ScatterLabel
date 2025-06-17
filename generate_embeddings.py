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
from PIL import Image
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    image_path: Optional[str] = None  # Path to the original image
    cropped_image_path: Optional[str] = None  # Path to the cropped image


def crop_and_save_image(
    bbox: BoundingBox,
    image_dir: str,
    output_dir: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> Optional[str]:
    """
    Crop image based on bounding box and save to disk.
    
    Args:
        bbox: BoundingBox instance with coordinates
        image_dir: Directory containing original images
        output_dir: Directory to save cropped images
        image_width: Original image width (for denormalization if bbox is normalized)
        image_height: Original image height (for denormalization if bbox is normalized)
        
    Returns:
        Path to cropped image, or None if failed
    """
    if not bbox.image_id:
        return None
    
    # Construct image path
    image_path = Path(image_dir) / bbox.image_id
    if not image_path.exists():
        # Try with common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = Path(image_dir) / f"{bbox.image_id}{ext}"
            if test_path.exists():
                image_path = test_path
                break
        else:
            print(f"Warning: Image not found: {bbox.image_id}")
            return None
    
    try:
        # Open image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Denormalize coordinates if needed
        if image_width and image_height:
            # Assume coordinates are normalized
            x_min = int(bbox.x_min * img_width)
            y_min = int(bbox.y_min * img_height)
            x_max = int(bbox.x_max * img_width)
            y_max = int(bbox.y_max * img_height)
        else:
            # Assume coordinates are in pixels
            x_min = int(bbox.x_min)
            y_min = int(bbox.y_min)
            x_max = int(bbox.x_max)
            y_max = int(bbox.y_max)
        
        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(x_min + 1, min(x_max, img_width))
        y_max = max(y_min + 1, min(y_max, img_height))
        
        # Crop image
        cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # Create filename with encoded bbox info
        # Format: {image_id}_{x_min}_{y_min}_{x_max}_{y_max}_{class_name}_{hash}.jpg
        bbox_str = f"{x_min}_{y_min}_{x_max}_{y_max}"
        class_safe = bbox.class_name.replace(" ", "_").replace("/", "_")
        
        # Add short hash for uniqueness
        hash_input = f"{bbox.image_id}_{bbox_str}_{class_safe}_{bbox.annotation_id or ''}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        filename = f"{Path(bbox.image_id).stem}_{bbox_str}_{class_safe}_{short_hash}.jpg"
        
        # Create output directory if needed
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save cropped image
        cropped.save(output_path, "JPEG", quality=90)
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error cropping image {bbox.image_id}: {e}")
        return None


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
    annotation_col: Optional[str] = None,
    image_path_col: Optional[str] = None
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
            annotation_id=str(row[annotation_col]) if annotation_col and annotation_col in row else None,
            image_path=row.get(image_path_col) if image_path_col and image_path_col in row else None
        )
        bounding_boxes.append(bbox)
    
    return bounding_boxes


def crop_all_bounding_boxes(
    bounding_boxes: List[BoundingBox],
    image_dir: str,
    output_dir: str,
    normalized_coords: bool = False
) -> List[BoundingBox]:
    """
    Crop images for all bounding boxes and update their cropped_image_path.
    
    Args:
        bounding_boxes: List of BoundingBox instances
        image_dir: Directory containing original images
        output_dir: Directory to save cropped images
        normalized_coords: Whether bbox coordinates are normalized (0-1) or in pixels
        
    Returns:
        Updated list of BoundingBox instances with cropped_image_path set
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Cropping {len(bounding_boxes)} bounding boxes...")
    
    for bbox in tqdm(bounding_boxes, desc="Cropping images"):
        if normalized_coords:
            # Will need to get image dimensions during cropping
            cropped_path = crop_and_save_image(
                bbox, image_dir, output_dir, image_width=1, image_height=1
            )
        else:
            cropped_path = crop_and_save_image(bbox, image_dir, output_dir)
        
        if cropped_path:
            bbox.cropped_image_path = cropped_path
    
    # Count successful crops
    successful_crops = sum(1 for bbox in bounding_boxes if bbox.cropped_image_path)
    print(f"Successfully cropped {successful_crops}/{len(bounding_boxes)} images")
    
    return bounding_boxes


def save_embeddings_for_scatterplot(
    embeddings: np.ndarray,
    bounding_boxes: List[BoundingBox],
    output_path: str,
    include_tsne: bool = True,
    include_umap: bool = False,
    save_csv: bool = True
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
        
        if bbox.cropped_image_path:
            metadata["cropped_image_path"] = bbox.cropped_image_path
        
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
    
    # Save CSV if requested
    if save_csv:
        csv_path = Path(output_path).with_suffix('.csv')
        
        # Prepare data for CSV
        csv_data = []
        for i, (bbox, metadata) in enumerate(zip(bounding_boxes, data["metadata"])):
            row = {
                'index': i,
                'class_name': bbox.class_name,
                'x_min': bbox.x_min,
                'y_min': bbox.y_min,
                'x_max': bbox.x_max,
                'y_max': bbox.y_max,
                'image_id': bbox.image_id,
                'cropped_image_path': bbox.cropped_image_path or ''
            }
            
            if bbox.confidence is not None:
                row['confidence'] = bbox.confidence
            
            if bbox.annotation_id is not None:
                row['annotation_id'] = bbox.annotation_id
            
            # Add embedding dimensions
            for j, emb_val in enumerate(embeddings[i]):
                row[f'embedding_{j}'] = emb_val
            
            # Add 2D projections if available
            if 'tsne' in data and i < len(data['tsne']):
                row['tsne_x'] = data['tsne'][i][0]
                row['tsne_y'] = data['tsne'][i][1]
            
            if 'umap' in data and i < len(data['umap']):
                row['umap_x'] = data['umap'][i][0]
                row['umap_y'] = data['umap'][i][1]
            
            csv_data.append(row)
        
        # Save to CSV
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")


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
    
    # Image cropping arguments
    parser.add_argument("--crop-images", action="store_true", help="Crop images for each bounding box")
    parser.add_argument("--image-dir", help="Directory containing original images")
    parser.add_argument("--crops-dir", default="crops", help="Directory to save cropped images (default: crops)")
    parser.add_argument("--normalized-coords", action="store_true", help="Bounding box coordinates are normalized (0-1)")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save results as CSV in addition to JSON")
    
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
    
    # Crop images if requested
    if args.crop_images:
        if not args.image_dir:
            parser.error("--image-dir is required when --crop-images is specified")
        
        bboxes = crop_all_bounding_boxes(
            bboxes,
            image_dir=args.image_dir,
            output_dir=args.crops_dir,
            normalized_coords=args.normalized_coords
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
        include_umap=args.umap,
        save_csv=args.save_csv
    )


if __name__ == "__main__":
    main()