#!/usr/bin/env python3
"""
Convert embeddings JSON to CSV format suitable for React frontend.
"""

import json
import pandas as pd
import argparse


def convert_embeddings_to_csv(input_json: str, output_csv: str) -> None:
    """
    Convert embeddings JSON to CSV format for React frontend.
    
    Args:
        input_json: Path to input JSON file with embeddings
        output_csv: Path to output CSV file
    """
    # Load the JSON data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Extract data
    embeddings = data.get('embeddings', [])
    metadata_list = data.get('metadata', [])
    tsne_coords = data.get('tsne', [])
    umap_coords = data.get('umap', [])
    
    # Create a list of records for the DataFrame
    records = []
    
    for i, metadata in enumerate(metadata_list):
        record = {
            'index': metadata['index'],
            'annotation_id': metadata.get('annotation_id', ''),
            'image_id': metadata.get('image_id', ''),
            'class_name': metadata['class_name'],
            'x_min': metadata['bbox']['x_min'],
            'y_min': metadata['bbox']['y_min'],
            'x_max': metadata['bbox']['x_max'],
            'y_max': metadata['bbox']['y_max'],
            'confidence': metadata.get('confidence', 1.0),
        }
        
        # Add 2D coordinates
        if tsne_coords and i < len(tsne_coords):
            record['tsne_x'] = tsne_coords[i][0]
            record['tsne_y'] = tsne_coords[i][1]
        
        if umap_coords and i < len(umap_coords):
            record['umap_x'] = umap_coords[i][0]
            record['umap_y'] = umap_coords[i][1]
        
        # Add high-dimensional embedding (first few dimensions for reference)
        if embeddings and i < len(embeddings):
            # Just store first 5 dimensions as example
            for j in range(min(5, len(embeddings[i]))):
                record[f'embedding_dim_{j}'] = embeddings[i][j]
        
        records.append(record)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"Converted {len(records)} records to {output_csv}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nSample data:")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(description="Convert embeddings JSON to CSV")
    parser.add_argument("--input", "-i", default="embeddings.json", help="Input JSON file")
    parser.add_argument("--output", "-o", default="embeddings_for_visualization.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    convert_embeddings_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()