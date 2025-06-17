#!/usr/bin/env python3
"""
Sample diverse crops from the dataset and generate embeddings for visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Optional
import sys
from tqdm import tqdm

def sample_diverse_crops(
    df: pd.DataFrame,
    n_samples: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample diverse crops from the dataset ensuring class balance and spatial diversity.
    
    Args:
        df: DataFrame with annotation data
        n_samples: Number of samples to select
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    np.random.seed(random_state)
    
    # Calculate additional features for diversity sampling
    df = df.copy()
    
    # Convert box coordinates to x_min, y_min, x_max, y_max format
    df['x_min'] = df['box_x1']
    df['y_min'] = df['box_y1']
    df['x_max'] = df['box_x1'] + df['box_width']
    df['y_max'] = df['box_y1'] + df['box_height']
    
    # Normalize coordinates
    df['norm_x_min'] = df['x_min'] / df['image_width']
    df['norm_y_min'] = df['y_min'] / df['image_height']
    df['norm_x_max'] = df['x_max'] / df['image_width']
    df['norm_y_max'] = df['y_max'] / df['image_height']
    
    # Calculate spatial features
    df['norm_area'] = (df['norm_x_max'] - df['norm_x_min']) * (df['norm_y_max'] - df['norm_y_min'])
    df['aspect_ratio'] = df['box_width'] / df['box_height']
    df['x_center'] = (df['norm_x_min'] + df['norm_x_max']) / 2
    df['y_center'] = (df['norm_y_min'] + df['norm_y_max']) / 2
    
    # Size categories
    df['size_category'] = pd.qcut(
        df['norm_area'], 
        q=[0, 0.01, 0.05, 0.2, 1.0], 
        labels=['tiny', 'small', 'medium', 'large']
    )
    
    # Aspect ratio categories
    df['aspect_category'] = pd.cut(
        df['aspect_ratio'], 
        bins=[0, 0.67, 1.5, float('inf')], 
        labels=['tall', 'square', 'wide']
    )
    
    # Position quadrants
    df['position_quadrant'] = df.apply(
        lambda row: f"{'top' if row['y_center'] < 0.5 else 'bottom'}-"
                   f"{'left' if row['x_center'] < 0.5 else 'right'}",
        axis=1
    )
    
    # Get unique classes
    classes = df['class_name'].unique()
    samples_per_class = max(1, n_samples // len(classes))
    
    sampled_dfs = []
    
    # Stratified sampling by class
    for class_name in tqdm(classes, desc="Sampling by class"):
        class_df = df[df['class_name'] == class_name]
        
        if len(class_df) <= samples_per_class:
            sampled_dfs.append(class_df)
        else:
            # Sample with diversity within each class
            # Try to get diverse sizes, aspects, and positions
            class_sample = pd.DataFrame()
            
            # Sample from different size categories
            for size_cat in ['tiny', 'small', 'medium', 'large']:
                size_df = class_df[class_df['size_category'] == size_cat]
                if len(size_df) > 0:
                    n_from_size = max(1, samples_per_class // 4)
                    class_sample = pd.concat([
                        class_sample,
                        size_df.sample(min(len(size_df), n_from_size), random_state=random_state)
                    ])
            
            # If we don't have enough samples, add more randomly
            if len(class_sample) < samples_per_class:
                remaining = samples_per_class - len(class_sample)
                remaining_df = class_df[~class_df.index.isin(class_sample.index)]
                if len(remaining_df) > 0:
                    class_sample = pd.concat([
                        class_sample,
                        remaining_df.sample(min(len(remaining_df), remaining), random_state=random_state)
                    ])
            
            sampled_dfs.append(class_sample.head(samples_per_class))
    
    # Combine all samples
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we have more than n_samples, randomly sample down
    if len(result_df) > n_samples:
        result_df = result_df.sample(n_samples, random_state=random_state)
    
    # Add image_id from frame_uri
    result_df['image_id'] = result_df['frame_uri'].apply(lambda x: Path(x).stem)
    
    print(f"\nSampling summary:")
    print(f"Total samples: {len(result_df)}")
    print(f"Classes represented: {result_df['class_name'].nunique()}")
    print(f"\nClass distribution:")
    print(result_df['class_name'].value_counts())
    
    return result_df


def prepare_for_embeddings(df: pd.DataFrame, output_path: str) -> None:
    """
    Prepare the sampled data for embedding generation.
    
    Args:
        df: Sampled DataFrame
        output_path: Path to save the prepared CSV
    """
    # Select columns needed for embedding generation
    columns_for_embedding = [
        'annotation_id',
        'image_id',
        'class_name',
        'x_min',
        'y_min', 
        'x_max',
        'y_max',
        'norm_x_min',
        'norm_y_min',
        'norm_x_max',
        'norm_y_max',
        'norm_area',
        'aspect_ratio',
        'x_center',
        'y_center',
        'size_category',
        'aspect_category',
        'position_quadrant',
        'frame_uri',
        'image_width',
        'image_height'
    ]
    
    # Add a confidence column (set to 1.0 as placeholder if not available)
    df['confidence'] = 1.0
    columns_for_embedding.append('confidence')
    
    # Save the prepared data
    output_df = df[columns_for_embedding].copy()
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved prepared data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample diverse crops and prepare for embedding generation")
    parser.add_argument("--input", "-i", default="labelled_datasets.csv", help="Input CSV file with annotations")
    parser.add_argument("--output", "-o", default="sampled_annotations.csv", help="Output CSV file for sampled data")
    parser.add_argument("--n-samples", "-n", type=int, default=100, help="Number of samples to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} annotations")
    
    # Sample diverse crops
    sampled_df = sample_diverse_crops(df, n_samples=args.n_samples, random_state=args.seed)
    
    # Prepare for embedding generation
    prepare_for_embeddings(sampled_df, args.output)
    
    print(f"\nNext step: Run embedding generation with:")
    print(f"python generate_embeddings.py --input {args.output} --output embeddings.json --tsne")


if __name__ == "__main__":
    main()