"""
Feature engineering for H3 hexagonal aggregation.

This module provides functions for aggregating geospatial data into H3 hexagons
and computing derived features for demand analysis.
"""

import pandas as pd
import numpy as np
import h3
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def hex_aggregate(df: pd.DataFrame, res: int) -> pd.DataFrame:
    """
    Aggregate points into H3 hexagons and compute features.
    
    Args:
        df: DataFrame with columns lat, lng, spd, azm, randomized_id
        res: H3 resolution (7 or 8 recommended)
        
    Returns:
        DataFrame with hex features: hex, lat, lng, polygon, n_points, n_devices,
        stop_share, speed_p50, speed_p90, heading_strength
    """
    logger.info(f"Aggregating {len(df):,} points into H3 resolution {res}")
    
    # Convert lat/lng to H3 indices
    df = df.copy()
    df['hex'] = df.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lng'], res), axis=1)
    
    # Aggregate by hex
    agg_data = []
    
    for hex_id, group in df.groupby('hex'):
        if len(group) == 0:
            continue
            
        # Basic counts
        n_points = len(group)
        n_devices = group['randomized_id'].nunique()
        
        # Speed statistics
        speeds = group['spd'].values
        stop_share = np.mean(speeds < 0.5)
        speed_p50 = np.median(speeds)
        speed_p90 = np.percentile(speeds, 90)
        
        # Heading concentration (circular statistics)
        headings_rad = np.radians(group['azm'].values)
        mean_sin = np.mean(np.sin(headings_rad))
        mean_cos = np.mean(np.cos(headings_rad))
        heading_strength = np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Get hex centroid and polygon
        lat, lng = h3.cell_to_latlng(hex_id)
        polygon = build_polygon(hex_id)
        
        agg_data.append({
            'hex': hex_id,
            'lat': lat,
            'lng': lng,
            'polygon': polygon,
            'n_points': n_points,
            'n_devices': n_devices,
            'stop_share': stop_share,
            'speed_p50': speed_p50,
            'speed_p90': speed_p90,
            'heading_strength': heading_strength
        })
    
    result_df = pd.DataFrame(agg_data)
    logger.info(f"Created {len(result_df):,} hexagons")
    
    return result_df


def build_polygon(hex_id: str) -> List[List[float]]:
    """
    Build GeoJSON-style polygon coordinates for an H3 hexagon.
    
    Args:
        hex_id: H3 cell identifier
        
    Returns:
        List of [lng, lat] coordinate pairs forming the hexagon boundary
    """
    try:
        # Get hex boundary coordinates
        boundary = h3.cell_to_boundary(hex_id)
        # Convert to list of [lng, lat] pairs
        polygon = [[coord[1], coord[0]] for coord in boundary]  # [lng, lat]
        return polygon
    except Exception as e:
        logger.warning(f"Failed to build polygon for hex {hex_id}: {e}")
        return []


def compute_hex_neighbors(hex_id: str, k: int = 1) -> List[str]:
    """
    Get k-ring neighbors of an H3 hexagon.
    
    Args:
        hex_id: H3 cell identifier
        k: Ring distance (1 = immediate neighbors)
        
    Returns:
        List of neighboring hex IDs
    """
    try:
        return h3.grid_disk(hex_id, k)
    except Exception as e:
        logger.warning(f"Failed to get neighbors for hex {hex_id}: {e}")
        return []


def get_hex_resolution(hex_id: str) -> int:
    """
    Get the resolution of an H3 hexagon.
    
    Args:
        hex_id: H3 cell identifier
        
    Returns:
        Resolution level
    """
    try:
        return h3.cell_to_res(hex_id)
    except Exception as e:
        logger.warning(f"Failed to get resolution for hex {hex_id}: {e}")
        return -1


def validate_hex_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean hex aggregation data.
    
    Args:
        df: DataFrame with hex features
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Validating {len(df):,} hex records")
    
    original_count = len(df)
    
    # Remove hexes with invalid data
    df = df.dropna(subset=['hex', 'lat', 'lng', 'n_points', 'n_devices'])
    
    # Remove hexes with zero points or devices
    df = df[(df['n_points'] > 0) & (df['n_devices'] > 0)]
    
    # Validate speed statistics
    df = df[(df['speed_p50'] >= 0) & (df['speed_p90'] >= 0)]
    
    # Validate heading strength (should be 0-1)
    df['heading_strength'] = df['heading_strength'].clip(0, 1)
    
    # Validate stop share (should be 0-1)
    df['stop_share'] = df['stop_share'].clip(0, 1)
    
    removed_count = original_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count:,} invalid hex records")
    
    logger.info(f"Validated {len(df):,} hex records")
    
    return df
