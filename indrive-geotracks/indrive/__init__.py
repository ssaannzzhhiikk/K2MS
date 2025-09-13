"""
inDrive Geotracks Analysis Library

A production-ready library for analyzing geospatial ride-hailing data
using H3 hexagonal indexing and advanced analytics.
"""

__version__ = "1.0.0"
__author__ = "inDrive Hackathon Team"

from .io import load_points_csv, ensure_csv_ext
from .features import hex_aggregate, build_polygon
from .demand import compute_demand
from .allocate import coverage_sets, greedy_select, baseline_topk
from .anomalies import neighbor_stats, zscore_speed, isolation_forest_flags
from .viz import make_map, add_hex_layer, add_markers

__all__ = [
    "load_points_csv",
    "ensure_csv_ext", 
    "hex_aggregate",
    "build_polygon",
    "compute_demand",
    "coverage_sets",
    "greedy_select",
    "baseline_topk",
    "neighbor_stats",
    "zscore_speed",
    "isolation_forest_flags",
    "make_map",
    "add_hex_layer",
    "add_markers",
]
