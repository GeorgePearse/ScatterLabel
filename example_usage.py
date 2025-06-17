#!/usr/bin/env python3
"""
Example usage of the embedding generator with different data formats.
"""

import pandas as pd
import numpy as np
from generate_embeddings import (
    JINAEmbeddingGenerator,
    BoundingBox,
    load_bounding_boxes_from_dataframe,
    save_embeddings_for_scatterplot
)


def example_coco_format():
    """Example with COCO-style annotations."""
    # COCO format typically has: category_id, bbox=[x, y, width, height]
    coco_data = pd.DataFrame({
        "category_name": ["person", "bicycle", "car", "motorcycle", "airplane"],
        "bbox_x": [100, 200, 300, 400, 500],
        "bbox_y": [100, 150, 200, 250, 300],
        "bbox_width": [50, 60, 80, 70, 100],
        "bbox_height": [100, 80, 60, 90, 50],
        "image_id": ["img_001", "img_001", "img_002", "img_002", "img_003"],
        "score": [0.99, 0.95, 0.88, 0.92, 0.97]
    })
    
    # Convert COCO bbox format to min/max format
    coco_data["x_min"] = coco_data["bbox_x"]
    coco_data["y_min"] = coco_data["bbox_y"]
    coco_data["x_max"] = coco_data["bbox_x"] + coco_data["bbox_width"]
    coco_data["y_max"] = coco_data["bbox_y"] + coco_data["bbox_height"]
    
    # Load bounding boxes
    bboxes = load_bounding_boxes_from_dataframe(
        coco_data,
        class_col="category_name",
        bbox_cols={
            "x_min": "x_min",
            "y_min": "y_min",
            "x_max": "x_max",
            "y_max": "y_max"
        },
        image_col="image_id",
        confidence_col="score"
    )
    
    # Generate embeddings
    generator = JINAEmbeddingGenerator()
    embeddings = generator.generate_embeddings(bboxes)
    
    # Save results
    save_embeddings_for_scatterplot(
        embeddings,
        bboxes,
        "coco_embeddings.json",
        include_tsne=True
    )


def example_yolo_format():
    """Example with YOLO-style annotations."""
    # YOLO format: class_id, center_x, center_y, width, height (normalized)
    # Assuming image dimensions are 640x480
    img_width, img_height = 640, 480
    
    yolo_data = pd.DataFrame({
        "class_name": ["person", "dog", "cat", "bird", "car"],
        "center_x_norm": [0.5, 0.3, 0.7, 0.2, 0.8],
        "center_y_norm": [0.5, 0.4, 0.6, 0.3, 0.7],
        "width_norm": [0.1, 0.15, 0.12, 0.08, 0.2],
        "height_norm": [0.2, 0.18, 0.15, 0.1, 0.15],
        "confidence": [0.95, 0.92, 0.88, 0.90, 0.94]
    })
    
    # Convert normalized YOLO format to pixel coordinates
    yolo_data["x_min"] = (yolo_data["center_x_norm"] - yolo_data["width_norm"]/2) * img_width
    yolo_data["y_min"] = (yolo_data["center_y_norm"] - yolo_data["height_norm"]/2) * img_height
    yolo_data["x_max"] = (yolo_data["center_x_norm"] + yolo_data["width_norm"]/2) * img_width
    yolo_data["y_max"] = (yolo_data["center_y_norm"] + yolo_data["height_norm"]/2) * img_height
    
    # Load and process
    bboxes = load_bounding_boxes_from_dataframe(
        yolo_data,
        class_col="class_name",
        confidence_col="confidence"
    )
    
    generator = JINAEmbeddingGenerator()
    embeddings = generator.generate_embeddings(bboxes)
    
    save_embeddings_for_scatterplot(
        embeddings,
        bboxes,
        "yolo_embeddings.json",
        include_tsne=True
    )


def example_custom_format():
    """Example with custom annotation format."""
    # Load from a CSV file with custom column names
    custom_data = pd.DataFrame({
        "object_type": ["traffic_light", "stop_sign", "pedestrian", "cyclist", "vehicle"],
        "left": [50, 150, 250, 350, 450],
        "top": [50, 100, 150, 200, 250],
        "right": [100, 200, 300, 400, 550],
        "bottom": [150, 180, 250, 280, 350],
        "frame_id": [1, 1, 2, 2, 3],
        "detection_score": [0.85, 0.90, 0.92, 0.88, 0.95]
    })
    
    # Custom mapping
    bboxes = load_bounding_boxes_from_dataframe(
        custom_data,
        class_col="object_type",
        bbox_cols={
            "x_min": "left",
            "y_min": "top",
            "x_max": "right",
            "y_max": "bottom"
        },
        image_col="frame_id",
        confidence_col="detection_score"
    )
    
    generator = JINAEmbeddingGenerator()
    embeddings = generator.generate_embeddings(bboxes)
    
    save_embeddings_for_scatterplot(
        embeddings,
        bboxes,
        "custom_embeddings.json",
        include_tsne=True,
        include_umap=True  # Try UMAP as well
    )


def example_batch_processing():
    """Example of processing a large dataset in batches."""
    # Generate a larger dataset
    n_objects = 1000
    classes = ["person", "car", "bicycle", "dog", "cat", "bird", "truck", "bus"]
    
    large_data = pd.DataFrame({
        "class_name": np.random.choice(classes, n_objects),
        "x_min": np.random.uniform(0, 800, n_objects),
        "y_min": np.random.uniform(0, 600, n_objects),
        "x_max": np.random.uniform(50, 200, n_objects),
        "y_max": np.random.uniform(50, 200, n_objects),
        "image_id": [f"img_{i//10:04d}" for i in range(n_objects)],
        "confidence": np.random.uniform(0.7, 1.0, n_objects)
    })
    
    # Make sure x_max > x_min and y_max > y_min
    large_data["x_max"] += large_data["x_min"]
    large_data["y_max"] += large_data["y_min"]
    
    # Process with custom batch size
    bboxes = load_bounding_boxes_from_dataframe(
        large_data,
        class_col="class_name",
        image_col="image_id",
        confidence_col="confidence"
    )
    
    generator = JINAEmbeddingGenerator()
    embeddings = generator.generate_embeddings(bboxes, batch_size=50)
    
    save_embeddings_for_scatterplot(
        embeddings,
        bboxes,
        "large_dataset_embeddings.json",
        include_tsne=True
    )
    
    print(f"Processed {len(bboxes)} bounding boxes")
    print(f"Embedding dimensions: {embeddings.shape}")


if __name__ == "__main__":
    print("Running COCO format example...")
    example_coco_format()
    
    print("\nRunning YOLO format example...")
    example_yolo_format()
    
    print("\nRunning custom format example...")
    example_custom_format()
    
    print("\nRunning batch processing example...")
    example_batch_processing()