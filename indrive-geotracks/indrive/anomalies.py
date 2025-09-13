"""
Anomaly detection for geospatial data.

This module provides functions for detecting anomalous patterns in hex-based
geospatial data using statistical and machine learning approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


def neighbor_stats(df: pd.DataFrame, res: int) -> pd.DataFrame:
    """
    Compute neighborhood statistics for each hex.
    
    Args:
        df: DataFrame with hex features
        res: H3 resolution level
        
    Returns:
        DataFrame with neighborhood statistics added
    """
    from .features import compute_hex_neighbors
    
    logger.info(f"Computing neighborhood stats for {len(df):,} hexes at resolution {res}")
    
    df = df.copy()
    
    # Create hex lookup for fast access
    hex_lookup = df.set_index('hex')
    
    # Compute neighborhood stats for each hex
    neighbor_stats_list = []
    
    for _, row in df.iterrows():
        hex_id = row['hex']
        
        try:
            # Get immediate neighbors (k-ring=1)
            neighbors = compute_hex_neighbors(hex_id, k=1)
            
            # Filter to neighbors that exist in our dataset
            valid_neighbors = [h for h in neighbors if h in hex_lookup.index]
            
            if len(valid_neighbors) == 0:
                # No valid neighbors, use self
                neighbor_stats = {
                    'neighbor_count': 0,
                    'neighbor_speed_p90_mean': row['speed_p90'],
                    'neighbor_speed_p90_std': 0.0,
                    'neighbor_heading_strength_mean': row['heading_strength'],
                    'neighbor_heading_strength_std': 0.0
                }
            else:
                # Compute neighbor statistics
                neighbor_data = hex_lookup.loc[valid_neighbors]
                
                neighbor_stats = {
                    'neighbor_count': len(valid_neighbors),
                    'neighbor_speed_p90_mean': neighbor_data['speed_p90'].mean(),
                    'neighbor_speed_p90_std': neighbor_data['speed_p90'].std(),
                    'neighbor_heading_strength_mean': neighbor_data['heading_strength'].mean(),
                    'neighbor_heading_strength_std': neighbor_data['heading_strength'].std()
                }
            
            neighbor_stats_list.append(neighbor_stats)
            
        except Exception as e:
            logger.warning(f"Failed to compute neighbor stats for hex {hex_id}: {e}")
            neighbor_stats_list.append({
                'neighbor_count': 0,
                'neighbor_speed_p90_mean': row['speed_p90'],
                'neighbor_speed_p90_std': 0.0,
                'neighbor_heading_strength_mean': row['heading_strength'],
                'neighbor_heading_strength_std': 0.0
            })
    
    # Add neighbor stats to dataframe
    neighbor_df = pd.DataFrame(neighbor_stats_list)
    for col in neighbor_df.columns:
        df[col] = neighbor_df[col]
    
    logger.info("Neighborhood statistics computed")
    
    return df


def zscore_speed(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag hexes with anomalous speed patterns using z-scores.
    
    Args:
        df: DataFrame with neighborhood statistics
        threshold: Z-score threshold for flagging anomalies
        
    Returns:
        DataFrame with anomaly flags added
    """
    logger.info(f"Computing speed z-scores with threshold {threshold}")
    
    df = df.copy()
    
    # Compute z-scores for speed_p90 relative to neighborhood
    df['speed_zscore'] = np.where(
        df['neighbor_speed_p90_std'] > 0,
        (df['speed_p90'] - df['neighbor_speed_p90_mean']) / df['neighbor_speed_p90_std'],
        0.0
    )
    
    # Flag anomalies
    df['speed_anomaly'] = np.abs(df['speed_zscore']) > threshold
    
    # Add anomaly reason
    def get_anomaly_reason(row):
        if row['speed_anomaly']:
            if row['speed_zscore'] > threshold:
                return f"High speed (z={row['speed_zscore']:.1f})"
            else:
                return f"Low speed (z={row['speed_zscore']:.1f})"
        else:
            return "Normal"
    
    df['anomaly_reason'] = df.apply(get_anomaly_reason, axis=1)
    
    anomaly_count = df['speed_anomaly'].sum()
    logger.info(f"Flagged {anomaly_count:,} speed anomalies ({anomaly_count/len(df)*100:.1f}%)")
    
    return df


def isolation_forest_flags(
    df: pd.DataFrame, 
    features: List[str] = None,
    contamination: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Flag global outliers using Isolation Forest.
    
    Args:
        df: DataFrame with hex features
        features: List of feature columns to use (default: speed and heading features)
        contamination: Expected proportion of outliers
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with isolation forest flags added
    """
    if features is None:
        features = ['speed_p50', 'speed_p90', 'stop_share', 'heading_strength']
    
    logger.info(f"Running Isolation Forest on {len(features)} features with contamination {contamination}")
    
    df = df.copy()
    
    # Prepare feature matrix
    feature_matrix = df[features].fillna(0).values
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    try:
        outlier_labels = iso_forest.fit_predict(feature_matrix)
        df['isolation_anomaly'] = outlier_labels == -1
        
        # Add anomaly score
        df['anomaly_score'] = iso_forest.score_samples(feature_matrix)
        
        anomaly_count = df['isolation_anomaly'].sum()
        logger.info(f"Flagged {anomaly_count:,} isolation anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Isolation Forest failed: {e}")
        df['isolation_anomaly'] = False
        df['anomaly_score'] = 0.0
    
    return df


def combined_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine different anomaly detection methods.
    
    Args:
        df: DataFrame with individual anomaly flags
        
    Returns:
        DataFrame with combined anomaly flags
    """
    logger.info("Combining anomaly detection methods")
    
    df = df.copy()
    
    # Combine flags
    df['any_anomaly'] = df.get('speed_anomaly', False) | df.get('isolation_anomaly', False)
    
    # Create combined reason
    reasons = []
    if 'speed_anomaly' in df.columns and df['speed_anomaly'].any():
        reasons.append("Speed")
    if 'isolation_anomaly' in df.columns and df['isolation_anomaly'].any():
        reasons.append("Isolation")
    
    df['combined_reason'] = df.apply(
        lambda row: " + ".join(reasons) if row['any_anomaly'] else "Normal",
        axis=1
    )
    
    anomaly_count = df['any_anomaly'].sum()
    logger.info(f"Total anomalies: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")
    
    return df


def get_anomaly_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for anomaly detection results.
    
    Args:
        df: DataFrame with anomaly flags
        
    Returns:
        Dictionary with anomaly summary statistics
    """
    summary = {
        'total_hexes': len(df),
        'speed_anomalies': df.get('speed_anomaly', pd.Series([False] * len(df))).sum(),
        'isolation_anomalies': df.get('isolation_anomaly', pd.Series([False] * len(df))).sum(),
        'any_anomalies': df.get('any_anomaly', pd.Series([False] * len(df))).sum()
    }
    
    # Add percentages
    for key in ['speed_anomalies', 'isolation_anomalies', 'any_anomalies']:
        summary[f'{key}_pct'] = summary[key] / summary['total_hexes'] * 100
    
    # Add feature statistics for anomalies
    if 'any_anomaly' in df.columns:
        anomaly_df = df[df['any_anomaly']]
        if len(anomaly_df) > 0:
            summary['anomaly_features'] = {
                'avg_speed_p90': anomaly_df['speed_p90'].mean(),
                'avg_heading_strength': anomaly_df['heading_strength'].mean(),
                'avg_demand': anomaly_df.get('demand', pd.Series([0] * len(anomaly_df))).mean()
            }
    
    return summary
