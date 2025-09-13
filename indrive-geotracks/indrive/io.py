"""
I/O utilities for loading and processing geospatial data.

This module provides safe, memory-efficient loading of large CSV files
with proper error handling and data validation.
"""

import os
import pandas as pd
from typing import Iterator, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_csv_ext(path: str) -> str:
    """
    Ensure the file path has a .csv extension.
    
    Args:
        path: File path that may or may not have .csv extension
        
    Returns:
        Path with .csv extension
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        # Try adding .csv extension
        csv_path = f"{path}.csv"
        if os.path.exists(csv_path):
            return csv_path
        raise FileNotFoundError(f"File not found: {path}")
    
    return path


def load_points_csv(
    path: str, 
    chunk_size: int = 200_000,
    astana_bbox: bool = True
) -> Iterator[pd.DataFrame]:
    """
    Load geospatial points CSV in chunks with Astana filtering.
    
    Args:
        path: Path to CSV file (with or without .csv extension)
        chunk_size: Number of rows per chunk
        astana_bbox: Whether to filter to Astana bounding box
        
    Yields:
        DataFrame chunks with columns: randomized_id, lat, lng, alt, spd, azm
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    path = ensure_csv_ext(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading CSV from {path} in chunks of {chunk_size:,}")
    
    # Define expected columns and dtypes
    expected_cols = ['randomized_id', 'lat', 'lng', 'alt', 'spd', 'azm']
    dtype_hints = {
        'randomized_id': 'str',
        'lat': 'float32',
        'lng': 'float32', 
        'alt': 'float32',
        'spd': 'float32',
        'azm': 'float32'
    }
    
    chunk_count = 0
    total_rows = 0
    
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=dtype_hints):
            chunk_count += 1
            
            # Validate required columns
            missing_cols = set(expected_cols) - set(chunk.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter to Astana bounding box if requested
            if astana_bbox:
                chunk = chunk[
                    (chunk['lat'] >= 51.0) & (chunk['lat'] <= 51.2) &
                    (chunk['lng'] >= 71.2) & (chunk['lng'] <= 71.6)
                ]
            
            # Drop rows with NaN values in critical columns
            chunk = chunk.dropna(subset=['lat', 'lng', 'spd', 'azm'])
            
            # Clean speed data
            chunk['spd'] = chunk['spd'].clip(lower=0)  # Remove negative speeds
            
            total_rows += len(chunk)
            
            if len(chunk) > 0:
                logger.debug(f"Chunk {chunk_count}: {len(chunk):,} rows")
                yield chunk
            else:
                logger.debug(f"Chunk {chunk_count}: empty after filtering")
                
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise
    
    logger.info(f"Loaded {total_rows:,} total rows in {chunk_count} chunks")


def get_file_info(path: str) -> dict:
    """
    Get basic information about a CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Dictionary with file info (size, estimated rows, etc.)
    """
    path = ensure_csv_ext(path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    file_size = os.path.getsize(path)
    
    # Estimate row count by reading first few lines
    with open(path, 'r') as f:
        first_line = f.readline()
        header_size = len(first_line.encode('utf-8'))
        
        # Sample a few lines to estimate average line size
        sample_lines = []
        for _ in range(10):
            line = f.readline()
            if line:
                sample_lines.append(len(line.encode('utf-8')))
        
        if sample_lines:
            avg_line_size = sum(sample_lines) / len(sample_lines)
            estimated_rows = int((file_size - header_size) / avg_line_size)
        else:
            estimated_rows = 0
    
    return {
        'path': path,
        'size_mb': file_size / (1024 * 1024),
        'estimated_rows': estimated_rows,
        'header_size': header_size
    }
