#!/usr/bin/env python3
"""
Precompute H3 hex features from geospatial data.

This script processes the raw geospatial CSV file and creates aggregated
hex features for efficient visualization and analysis.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from indrive.io import load_points_csv, get_file_info
from indrive.features import hex_aggregate, validate_hex_data
from indrive.demand import compute_demand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_chunks(input_path: str, resolutions: List[int]) -> dict:
    """
    Process CSV file in chunks and aggregate to hex features.
    
    Args:
        input_path: Path to input CSV file
        resolutions: List of H3 resolutions to compute
        
    Returns:
        Dictionary mapping resolution to aggregated DataFrame
    """
    logger.info(f"Processing {input_path} for resolutions {resolutions}")
    
    # Get file info
    file_info = get_file_info(input_path)
    logger.info(f"File size: {file_info['size_mb']:.1f} MB, estimated rows: {file_info['estimated_rows']:,}")
    
    # Initialize aggregation dictionaries
    hex_data = {res: [] for res in resolutions}
    
    # Process in chunks
    chunk_count = 0
    total_points = 0
    
    for chunk in load_points_csv(input_path, chunk_size=200_000):
        chunk_count += 1
        total_points += len(chunk)
        
        logger.info(f"Processing chunk {chunk_count}: {len(chunk):,} points")
        
        # Winsorize speeds at p99 to remove extreme noise
        speed_p99 = chunk['spd'].quantile(0.99)
        chunk['spd'] = chunk['spd'].clip(upper=speed_p99)
        
        # Aggregate for each resolution
        for res in resolutions:
            try:
                hex_df = hex_aggregate(chunk, res)
                if len(hex_df) > 0:
                    hex_data[res].append(hex_df)
                    logger.debug(f"Resolution {res}: {len(hex_df):,} hexes")
            except Exception as e:
                logger.error(f"Failed to aggregate chunk {chunk_count} at resolution {res}: {e}")
                continue
    
    logger.info(f"Processed {total_points:,} total points in {chunk_count} chunks")
    
    # Combine all chunks for each resolution
    final_data = {}
    for res in resolutions:
        if hex_data[res]:
            logger.info(f"Combining {len(hex_data[res])} chunks for resolution {res}")
            
            # Concatenate all chunks
            combined_df = pd.concat(hex_data[res], ignore_index=True)
            
            # Group by hex and sum numeric columns
            numeric_cols = ['n_points', 'n_devices']
            agg_dict = {col: 'sum' for col in numeric_cols}
            
            # For other columns, take the mean
            other_cols = ['stop_share', 'speed_p50', 'speed_p90', 'heading_strength']
            for col in other_cols:
                if col in combined_df.columns:
                    agg_dict[col] = 'mean'
            
            # Aggregate by hex
            final_df = combined_df.groupby('hex').agg(agg_dict).reset_index()
            
            # Add back lat, lng, polygon from first occurrence
            first_occurrence = combined_df.groupby('hex').first()
            final_df['lat'] = final_df['hex'].map(first_occurrence['lat'])
            final_df['lng'] = final_df['hex'].map(first_occurrence['lng'])
            final_df['polygon'] = final_df['hex'].map(first_occurrence['polygon'])
            
            # Validate and clean data
            final_df = validate_hex_data(final_df)
            
            # Compute demand scores
            final_df = compute_demand(final_df)
            
            final_data[res] = final_df
            logger.info(f"Resolution {res}: {len(final_df):,} final hexes")
        else:
            logger.warning(f"No data for resolution {res}")
            final_data[res] = pd.DataFrame()
    
    return final_data


def save_artifacts(data: dict, output_dir: str):
    """
    Save aggregated data as Parquet files.
    
    Args:
        data: Dictionary mapping resolution to DataFrame
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for res, df in data.items():
        if len(df) == 0:
            logger.warning(f"No data to save for resolution {res}")
            continue
        
        output_path = os.path.join(output_dir, f'hex_res{res}.parquet')
        
        # Select and order columns
        columns = [
            'hex', 'lat', 'lng', 'polygon', 'n_points', 'n_devices', 
            'stop_share', 'speed_p50', 'speed_p90', 'heading_strength', 'demand'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in columns if col in df.columns]
        df_to_save = df[available_columns].copy()
        
        # Save as Parquet
        df_to_save.to_parquet(output_path, index=False)
        
        logger.info(f"Saved {len(df_to_save):,} hexes to {output_path}")
        
        # Log summary statistics
        logger.info(f"Resolution {res} summary:")
        logger.info(f"  - Hexes: {len(df_to_save):,}")
        logger.info(f"  - Total points: {df_to_save['n_points'].sum():,}")
        logger.info(f"  - Total devices: {df_to_save['n_devices'].sum():,}")
        logger.info(f"  - Demand range: {df_to_save['demand'].min():.3f} - {df_to_save['demand'].max():.3f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Precompute H3 hex features from geospatial data')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--out', required=True, help='Output directory for artifacts')
    parser.add_argument('--res', nargs='+', type=int, default=[7, 8], 
                       help='H3 resolutions to compute (default: 7 8)')
    parser.add_argument('--chunk-size', type=int, default=200_000,
                       help='Chunk size for processing (default: 200000)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Validate resolutions
    if not all(1 <= res <= 15 for res in args.res):
        logger.error("H3 resolutions must be between 1 and 15")
        sys.exit(1)
    
    logger.info(f"Starting preprocessing with resolutions: {args.res}")
    
    try:
        # Process data
        hex_data = process_chunks(args.input, args.res)
        
        # Save artifacts
        save_artifacts(hex_data, args.out)
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
