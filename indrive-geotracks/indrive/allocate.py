"""
Driver allocation and coverage optimization algorithms.

This module provides greedy and baseline algorithms for selecting optimal
staging locations to maximize demand coverage.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def coverage_sets(hexes: List[str], R: int = 1) -> Dict[str, Set[str]]:
    """
    Compute coverage sets for each hex (hexes within k-ring R).
    
    Args:
        hexes: List of H3 hex identifiers
        R: Coverage radius (k-ring distance)
        
    Returns:
        Dictionary mapping each hex to its coverage set
    """
    from .features import compute_hex_neighbors
    
    coverage = {}
    
    for hex_id in hexes:
        try:
            # Get all hexes within k-ring R (including the hex itself)
            neighbors = set()
            for k in range(R + 1):
                ring_hexes = compute_hex_neighbors(hex_id, k)
                neighbors.update(ring_hexes)
            
            coverage[hex_id] = neighbors
        except Exception as e:
            logger.warning(f"Failed to compute coverage for hex {hex_id}: {e}")
            coverage[hex_id] = {hex_id}  # At least cover itself
    
    return coverage


def greedy_select(df: pd.DataFrame, k: int, R: int = 1, min_devices: int = 1) -> List[str]:
    """
    Greedy algorithm for selecting k staging hexes to maximize covered demand.
    
    Args:
        df: DataFrame with hex features and demand scores
        k: Number of staging locations to select
        R: Coverage radius (k-ring distance)
        min_devices: Minimum devices required for a hex to be considered
        
    Returns:
        List of selected hex IDs in order of selection
    """
    logger.info(f"Running greedy selection: k={k}, R={R}, min_devices={min_devices}")
    
    # Filter hexes by minimum devices
    candidate_hexes = df[df['n_devices'] >= min_devices]['hex'].tolist()
    
    if len(candidate_hexes) == 0:
        logger.warning("No hexes meet minimum device requirement")
        return []
    
    if k >= len(candidate_hexes):
        logger.info(f"Requested {k} hexes but only {len(candidate_hexes)} available")
        return candidate_hexes
    
    # Compute coverage sets for all candidates
    coverage_sets_dict = coverage_sets(candidate_hexes, R)
    
    # Create demand lookup
    demand_lookup = df.set_index('hex')['demand'].to_dict()
    
    # Greedy selection
    selected = []
    covered_hexes = set()
    total_covered_demand = 0.0
    
    for i in range(k):
        best_hex = None
        best_new_demand = 0.0
        
        for hex_id in candidate_hexes:
            if hex_id in selected:
                continue
            
            # Compute newly covered demand
            hex_coverage = coverage_sets_dict[hex_id]
            new_hexes = hex_coverage - covered_hexes
            
            new_demand = sum(
                demand_lookup.get(h, 0) for h in new_hexes
            )
            
            if new_demand > best_new_demand:
                best_new_demand = new_demand
                best_hex = hex_id
        
        if best_hex is None:
            logger.warning(f"Could not find hex {i+1}/{k}")
            break
        
        # Add best hex to selection
        selected.append(best_hex)
        covered_hexes.update(coverage_sets_dict[best_hex])
        total_covered_demand += best_new_demand
        
        logger.debug(f"Selected hex {i+1}/{k}: {best_hex} (new demand: {best_new_demand:.3f})")
    
    logger.info(f"Greedy selection complete: {len(selected)} hexes, {len(covered_hexes)} covered, {total_covered_demand:.3f} total demand")
    
    return selected


def baseline_topk(df: pd.DataFrame, k: int, min_devices: int = 1) -> List[str]:
    """
    Baseline algorithm: select top-k hexes by demand score.
    
    Args:
        df: DataFrame with hex features and demand scores
        k: Number of staging locations to select
        min_devices: Minimum devices required for a hex to be considered
        
    Returns:
        List of selected hex IDs ordered by demand (descending)
    """
    logger.info(f"Running baseline selection: k={k}, min_devices={min_devices}")
    
    # Filter and sort by demand
    filtered_df = df[df['n_devices'] >= min_devices].copy()
    filtered_df = filtered_df.sort_values('demand', ascending=False)
    
    selected = filtered_df.head(k)['hex'].tolist()
    
    logger.info(f"Baseline selection complete: {len(selected)} hexes")
    
    return selected


def compute_coverage_metrics(df: pd.DataFrame, selected_hexes: List[str], R: int = 1) -> Dict[str, float]:
    """
    Compute coverage metrics for selected staging hexes.
    
    Args:
        df: DataFrame with hex features and demand scores
        selected_hexes: List of selected hex IDs
        R: Coverage radius (k-ring distance)
        
    Returns:
        Dictionary with coverage metrics
    """
    if not selected_hexes:
        return {
            'total_hexes': 0,
            'covered_hexes': 0,
            'coverage_pct': 0.0,
            'total_demand': 0.0,
            'covered_demand': 0.0,
            'demand_coverage_pct': 0.0
        }
    
    # Get all hexes in dataset
    all_hexes = set(df['hex'].tolist())
    
    # Compute coverage
    coverage_sets_dict = coverage_sets(selected_hexes, R)
    covered_hexes = set()
    for hex_id in selected_hexes:
        covered_hexes.update(coverage_sets_dict.get(hex_id, {hex_id}))
    
    # Filter to only include hexes that exist in our dataset
    covered_hexes = covered_hexes.intersection(all_hexes)
    
    # Compute demand metrics
    total_demand = df['demand'].sum()
    covered_demand = df[df['hex'].isin(covered_hexes)]['demand'].sum()
    
    metrics = {
        'total_hexes': len(all_hexes),
        'covered_hexes': len(covered_hexes),
        'coverage_pct': len(covered_hexes) / len(all_hexes) * 100,
        'total_demand': total_demand,
        'covered_demand': covered_demand,
        'demand_coverage_pct': covered_demand / total_demand * 100 if total_demand > 0 else 0.0
    }
    
    return metrics


def compare_strategies(df: pd.DataFrame, k: int, R: int = 1, min_devices: int = 1) -> Dict[str, Any]:
    """
    Compare greedy vs baseline allocation strategies.
    
    Args:
        df: DataFrame with hex features and demand scores
        k: Number of staging locations to select
        R: Coverage radius (k-ring distance)
        min_devices: Minimum devices required for a hex to be considered
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing strategies: k={k}, R={R}, min_devices={min_devices}")
    
    # Run both strategies
    baseline_hexes = baseline_topk(df, k, min_devices)
    greedy_hexes = greedy_select(df, k, R, min_devices)
    
    # Compute metrics for both
    baseline_metrics = compute_coverage_metrics(df, baseline_hexes, R)
    greedy_metrics = compute_coverage_metrics(df, greedy_hexes, R)
    
    # Compute uplift
    demand_uplift = greedy_metrics['demand_coverage_pct'] - baseline_metrics['demand_coverage_pct']
    coverage_uplift = greedy_metrics['coverage_pct'] - baseline_metrics['coverage_pct']
    
    comparison = {
        'baseline': {
            'hexes': baseline_hexes,
            'metrics': baseline_metrics
        },
        'greedy': {
            'hexes': greedy_hexes,
            'metrics': greedy_metrics
        },
        'uplift': {
            'demand_coverage_pct': demand_uplift,
            'coverage_pct': coverage_uplift
        }
    }
    
    logger.info(f"Comparison complete - Greedy uplift: {demand_uplift:.1f}% demand, {coverage_uplift:.1f}% coverage")
    
    return comparison
