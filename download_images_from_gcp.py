#!/usr/bin/env python3
"""Download images from GCP Storage to local directory."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from google.cloud import storage
from tqdm import tqdm


def extract_unique_image_ids(embeddings_file: str) -> Set[str]:
    """Extract unique image IDs from embeddings file."""
    with open(embeddings_file, 'r') as f:
        data = json.load(f)
    
    image_ids = set()
    if 'metadata' in data:
        for item in data['metadata']:
            if 'image_id' in item:
                image_ids.add(item['image_id'])
    
    return image_ids


def download_image(
    client: storage.Client,
    bucket_name: str,
    image_id: str,
    source_prefix: str,
    output_dir: Path,
    file_extension: str = '.jpg'
) -> Optional[str]:
    """Download a single image from GCP Storage."""
    try:
        bucket = client.bucket(bucket_name)
        
        # Try different possible paths
        possible_paths = [
            f"{source_prefix}/{image_id}{file_extension}",
            f"{source_prefix}/{image_id}.jpeg",
            f"{source_prefix}/{image_id}.png",
            f"{image_id}{file_extension}",
            f"{image_id}.jpeg",
            f"{image_id}.png",
        ]
        
        for blob_path in possible_paths:
            blob = bucket.blob(blob_path)
            if blob.exists():
                output_path = output_dir / f"{image_id}.jpg"
                blob.download_to_filename(str(output_path))
                return str(output_path)
        
        return None
    except Exception as e:
        print(f"Error downloading {image_id}: {str(e)}")
        return None


def download_images_from_gcp(
    embeddings_file: str,
    bucket_name: str,
    source_prefix: str = "",
    output_dir: str = "./images",
    max_workers: int = 10
) -> None:
    """Download all images referenced in embeddings file from GCP Storage."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract unique image IDs
    print("Extracting image IDs from embeddings file...")
    image_ids = extract_unique_image_ids(embeddings_file)
    print(f"Found {len(image_ids)} unique images to download")
    
    if not image_ids:
        print("No image IDs found in embeddings file")
        return
    
    # Initialize GCP client
    client = storage.Client()
    
    # Download images in parallel
    successful_downloads = 0
    failed_downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_image_id = {
            executor.submit(
                download_image,
                client,
                bucket_name,
                image_id,
                source_prefix,
                output_path
            ): image_id
            for image_id in image_ids
        }
        
        # Process completed downloads
        with tqdm(total=len(image_ids), desc="Downloading images") as pbar:
            for future in as_completed(future_to_image_id):
                image_id = future_to_image_id[future]
                try:
                    result = future.result()
                    if result:
                        successful_downloads += 1
                    else:
                        failed_downloads.append(image_id)
                except Exception as e:
                    print(f"Exception for {image_id}: {str(e)}")
                    failed_downloads.append(image_id)
                pbar.update(1)
    
    # Print summary
    print(f"\nDownload complete:")
    print(f"  Successfully downloaded: {successful_downloads}/{len(image_ids)}")
    if failed_downloads:
        print(f"  Failed downloads: {len(failed_downloads)}")
        print(f"  First 10 failed IDs: {failed_downloads[:10]}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python download_images_from_gcp.py <embeddings_file> <bucket_name> [source_prefix] [output_dir]")
        print("\nExample:")
        print("  python download_images_from_gcp.py embeddings_with_ids.json my-bucket-name images ./images")
        sys.exit(1)
    
    embeddings_file = sys.argv[1]
    bucket_name = sys.argv[2]
    source_prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "./images"
    
    # Check if embeddings file exists
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file '{embeddings_file}' not found")
        sys.exit(1)
    
    # Set up GCP authentication if needed
    # You can set GOOGLE_APPLICATION_CREDENTIALS environment variable
    # or use gcloud auth application-default login
    
    download_images_from_gcp(
        embeddings_file=embeddings_file,
        bucket_name=bucket_name,
        source_prefix=source_prefix,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()