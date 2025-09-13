"""
Demand scoring and normalization utilities.

This module provides functions for computing normalized demand scores
and other metrics for hex-based analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_demand(df: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
    """
    Compute normalized demand scores for hexagons.
    
    Args:
        df: DataFrame with hex features (n_points, n_devices, stop_share, etc.)
        weights: Weights for demand components (default: 0.5 devices, 0.3 points, 0.2 stops)
        
    Returns:
        DataFrame with normalized features and demand scores
    """
    if weights is None:
        weights = {
            'n_devices': 0.5,
            'n_points': 0.3, 
            'stop_share': 0.2
        }
    
    logger.info(f"Computing demand scores for {len(df):,} hexes")
    
    df = df.copy()
    
    # Normalize features to [0, 1] using min-max scaling
    df['n_devices_norm'] = min_max_normalize(df['n_devices'])
    df['n_points_norm'] = min_max_normalize(df['n_points'])
    df['stop_share_norm'] = min_max_normalize(df['stop_share'])
    
    # Compute weighted demand score
    df['demand'] = (
        weights['n_devices'] * df['n_devices_norm'] +
        weights['n_points'] * df['n_points_norm'] +
        weights['stop_share'] * df['stop_share_norm']
    )
    
    # Ensure demand is in [0, 1]
    df['demand'] = df['demand'].clip(0, 1)
    
    logger.info(f"Demand score range: {df['demand'].min():.3f} - {df['demand'].max():.3f}")
    
    return df


def min_max_normalize(series: pd.Series, min_val: float = None, max_val: float = None) -> pd.Series:
    """
    Min-max normalization to [0, 1] range.
    
    Args:
        series: Input series to normalize
        min_val: Override minimum value (uses series min if None)
        max_val: Override maximum value (uses series max if None)
        
    Returns:
        Normalized series
    """
    if min_val is None:
        min_val = series.min()
    if max_val is None:
        max_val = series.max()
    
    if max_val == min_val:
        # Handle case where all values are the same
        return pd.Series(0.5, index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def compute_robust_normalization(series: pd.Series, percentile_range: Tuple[float, float] = (1, 99)) -> pd.Series:
    """
    Robust normalization using percentiles to handle outliers.
    
    Args:
        series: Input series to normalize
        percentile_range: Tuple of (lower, upper) percentiles for clipping
        
    Returns:
        Normalized series
    """
    lower_pct, upper_pct = percentile_range
    
    # Clip outliers using percentiles
    clipped = series.clip(
        lower=series.quantile(lower_pct / 100),
        upper=series.quantile(upper_pct / 100)
    )
    
    return min_max_normalize(clipped)


def compute_demand_components(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute detailed demand component statistics.
    
    Args:
        df: DataFrame with demand features
        
    Returns:
        Dictionary with component statistics
    """
    components = {
        'n_devices': {
            'min': df['n_devices'].min(),
            'max': df['n_devices'].max(),
            'mean': df['n_devices'].mean(),
            'median': df['n_devices'].median(),
            'std': df['n_devices'].std()
        },
        'n_points': {
            'min': df['n_points'].min(),
            'max': df['n_points'].max(),
            'mean': df['n_points'].mean(),
            'median': df['n_points'].median(),
            'std': df['n_points'].std()
        },
        'stop_share': {
            'min': df['stop_share'].min(),
            'max': df['stop_share'].max(),
            'mean': df['stop_share'].mean(),
            'median': df['stop_share'].median(),
            'std': df['stop_share'].std()
        },
        'demand': {
            'min': df['demand'].min(),
            'max': df['demand'].max(),
            'mean': df['demand'].mean(),
            'median': df['demand'].median(),
            'std': df['demand'].std()
        }
    }
    
    return components


def get_demand_quantiles(df: pd.DataFrame, quantiles: List[float] = None) -> Dict[str, float]:
    """
    Get demand score quantiles for visualization thresholds.
    
    Args:
        df: DataFrame with demand scores
        quantiles: List of quantiles to compute (default: [0.1, 0.25, 0.5, 0.75, 0.9])
        
    Returns:
        Dictionary mapping quantile names to values
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    result = {}
    for q in quantiles:
        result[f'q{int(q*100)}'] = df['demand'].quantile(q)
    
    return result


def filter_by_demand(df: pd.DataFrame, min_demand: float = 0.0, max_demand: float = 1.0) -> pd.DataFrame:
    """
    Filter hexes by demand score range.
    
    Args:
        df: DataFrame with demand scores
        min_demand: Minimum demand score (inclusive)
        max_demand: Maximum demand score (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    mask = (df['demand'] >= min_demand) & (df['demand'] <= max_demand)
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtered to {len(filtered_df):,} hexes (demand {min_demand:.2f}-{max_demand:.2f})")
    
    return filtered_df