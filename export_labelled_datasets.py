#!/usr/bin/env python3
"""Export machine_learning.labelled_datasets table to CSV."""

import os
from typing import Optional

import pandas as pd
import psycopg
from dotenv import load_dotenv
from tqdm import tqdm


class DatabaseExporter:
    """Handles exporting data from PostgreSQL to CSV."""
    
    def __init__(self, connection_string: str) -> None:
        """Initialize the exporter with a database connection string.
        
        Args:
            connection_string: PostgreSQL connection URI
        """
        self.connection_string = connection_string
    
    def export_table_to_csv(
        self, 
        schema: str, 
        table: str, 
        output_path: str,
        chunk_size: int = 10000
    ) -> None:
        """Export a PostgreSQL table to CSV with progress tracking.
        
        Args:
            schema: Database schema name
            table: Table name
            output_path: Path where CSV will be saved
            chunk_size: Number of rows to fetch at a time
        """
        with psycopg.connect(self.connection_string) as conn:
            # First, get the total row count
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                result = cur.fetchone()
                total_rows = result[0] if result else 0
                print(f"Total rows to export: {total_rows:,}")
            
            # Export data in chunks with progress bar
            query = f"SELECT * FROM {schema}.{table}"
            
            with conn.cursor(name='export_cursor') as cur:
                cur.execute(query)
                cur.itersize = chunk_size
                
                # Get column names
                columns = [desc[0] for desc in cur.description] if cur.description else []
                
                # Initialize progress bar
                pbar = tqdm(total=total_rows, desc="Exporting rows")
                
                # Write data in chunks
                first_chunk = True
                rows_written = 0
                
                while True:
                    rows = cur.fetchmany(chunk_size)
                    if not rows:
                        break
                    
                    df = pd.DataFrame(rows, columns=columns)
                    
                    # Write header only for first chunk
                    df.to_csv(
                        output_path,
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False
                    )
                    
                    first_chunk = False
                    rows_written += len(rows)
                    pbar.update(len(rows))
                
                pbar.close()
                print(f"Successfully exported {rows_written:,} rows to {output_path}")


def main() -> None:
    """Main function to export labelled_datasets table."""
    # Load environment variables
    load_dotenv()
    
    # Get database connection from environment
    postgres_uri = os.environ.get('POSTGRES_URI')
    if not postgres_uri:
        raise ValueError("POSTGRES_URI environment variable not set")
    
    # Initialize exporter
    exporter = DatabaseExporter(postgres_uri)
    
    # Export the table
    output_path = "labelled_datasets.csv"
    exporter.export_table_to_csv(
        schema="machine_learning",
        table="labelled_datasets",
        output_path=output_path
    )


if __name__ == "__main__":
    main()