#!/usr/bin/env python3
"""
Prepare CMR annotations from labelled_datasets.csv for embeddings generation.
"""

import pandas as pd
from pathlib import Path

# Read the labelled datasets
print("Loading labelled_datasets.csv...")
df = pd.read_csv('labelled_datasets.csv')

# Filter for CMR dataset
print("Filtering for CMR dataset...")
cmr_df = df[df['dataset_name'] == 'cmr'].copy()
print(f"Found {len(cmr_df)} CMR annotations")

# Transform to the format expected by generate_embeddings.py
# Convert box coordinates to x_min, y_min, x_max, y_max
cmr_df['x_min'] = cmr_df['box_x1']
cmr_df['y_min'] = cmr_df['box_y1']
cmr_df['x_max'] = cmr_df['box_x1'] + cmr_df['box_width']
cmr_df['y_max'] = cmr_df['box_y1'] + cmr_df['box_height']

# Extract image_id from frame_uri
# Example: gs://ingestion_prod/frames/2024-09-03T20:00:10.jpg -> 2024-09-03T20:00:10.jpg
cmr_df['image_id'] = cmr_df['frame_uri'].apply(lambda x: Path(x).name)

# Select relevant columns
output_df = cmr_df[['class_name', 'x_min', 'y_min', 'x_max', 'y_max', 'image_id', 'annotation_id']].copy()

# Save to CSV
output_path = 'cmr_annotations.csv'
output_df.to_csv(output_path, index=False)
print(f"Saved {len(output_df)} CMR annotations to {output_path}")

# Print some statistics
print("\nDataset statistics:")
print(f"Unique classes: {output_df['class_name'].nunique()}")
print(f"Unique images: {output_df['image_id'].nunique()}")
print("\nClass distribution:")
print(output_df['class_name'].value_counts())