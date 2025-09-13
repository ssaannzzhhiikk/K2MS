"""
Visualization utilities for Folium maps.

This module provides helper functions for creating and styling
interactive maps with H3 hexagons and markers.
"""

import folium
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Astana coordinates
ASTANA_CENTER = [51.1694, 71.4491]
ASTANA_ZOOM = 11


def make_map(center: List[float] = None, zoom: int = None) -> folium.Map:
    """
    Create a base Folium map centered on Astana.
    
    Args:
        center: [lat, lng] coordinates for map center
        zoom: Initial zoom level
        
    Returns:
        Folium Map object
    """
    if center is None:
        center = ASTANA_CENTER
    if zoom is None:
        zoom = ASTANA_ZOOM
    
    logger.info(f"Creating map centered at {center} with zoom {zoom}")
    
    # Create map with appropriate tiles
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add alternative tile layers
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
        attr='OpenStreetMap HOT',
        name='OpenStreetMap HOT',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB positron',
        name='CartoDB Positron',
        overlay=False,
        control=True
    ).add_to(m)
    
    return m


def get_color_scale(metric: str) -> Tuple[str, List[float], List[str]]:
    """
    Get color scale configuration for different metrics.
    
    Args:
        metric: Metric name ('demand', 'n_devices', 'stop_share', 'heading_strength', 'speed_p90')
        
    Returns:
        Tuple of (colormap_name, bins, colors)
    """
    color_scales = {
        'demand': ('YlOrRd', [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                  ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c']),
        'n_devices': ('Blues', [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6']),
        'stop_share': ('Greens', [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476']),
        'heading_strength': ('Purples', [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8']),
        'speed_p90': ('Reds', [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a'])
    }
    
    return color_scales.get(metric, color_scales['demand'])


def add_hex_layer(
    map_obj: folium.Map,
    df: pd.DataFrame,
    metric: str = 'demand',
    min_devices: int = 1,
    show_legend: bool = True
) -> folium.Map:
    """
    Add H3 hexagons as a GeoJSON layer to the map.
    
    Args:
        map_obj: Folium Map object
        df: DataFrame with hex features and polygons
        metric: Metric to color by
        min_devices: Minimum devices filter
        show_legend: Whether to show color legend
        
    Returns:
        Updated Folium Map object
    """
    logger.info(f"Adding hex layer: {metric}, min_devices={min_devices}")
    
    # Filter by minimum devices
    filtered_df = df[df['n_devices'] >= min_devices].copy()
    
    if len(filtered_df) == 0:
        logger.warning("No hexes meet minimum device requirement")
        return map_obj
    
    # Get color scale
    colormap_name, bins, colors = get_color_scale(metric)
    
    # Normalize metric values to [0, 1] for color mapping
    metric_values = filtered_df[metric].values
    if metric_values.max() > metric_values.min():
        normalized_values = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min())
    else:
        normalized_values = np.ones_like(metric_values) * 0.5
    
    # Create GeoJSON features
    features = []
    for _, row in filtered_df.iterrows():
        polygon = row['polygon']
        # Check if polygon is valid
        try:
            if (polygon is None or 
                not hasattr(polygon, '__len__') or 
                len(polygon) < 3):
                continue
        except (ValueError, TypeError):
            continue
        
        # Determine color based on normalized value
        color_idx = min(int(normalized_values[filtered_df.index.get_loc(row.name)] * (len(colors) - 1)), len(colors) - 1)
        color = colors[color_idx]
        
        # Convert polygon to GeoJSON format
        polygon_coords = []
        if hasattr(polygon, 'tolist'):
            # Convert numpy array to list
            polygon_list = polygon.tolist()
        else:
            polygon_list = list(polygon)
        
        # Convert to [lng, lat] format
        for coord in polygon_list:
            if len(coord) >= 2:
                polygon_coords.append([float(coord[1]), float(coord[0])])  # [lng, lat]
        
        # Create polygon feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            },
            "properties": {
                "hex": row['hex'],
                "lat": row['lat'],
                "lng": row['lng'],
                "n_points": int(row['n_points']),
                "n_devices": int(row['n_devices']),
                "stop_share": float(row['stop_share']),
                "speed_p50": float(row['speed_p50']),
                "speed_p90": float(row['speed_p90']),
                "heading_strength": float(row['heading_strength']),
                "demand": float(row.get('demand', 0)),
                "metric_value": float(row[metric]),
                "color": color
            }
        }
        features.append(feature)
    
    # Create GeoJSON layer
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Add to map
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
            'opacity': 0.8
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['hex', 'n_devices', 'n_points', 'demand', 'speed_p90', 'heading_strength'],
            aliases=['Hex ID', 'Devices', 'Points', 'Demand', 'Speed P90', 'Heading Strength'],
            localize=True,
            sticky=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['hex', 'n_devices', 'n_points', 'stop_share', 'speed_p50', 'speed_p90', 'heading_strength', 'demand'],
            aliases=['Hex ID', 'Devices', 'Points', 'Stop Share', 'Speed P50', 'Speed P90', 'Heading Strength', 'Demand'],
            localize=True,
            labels=True
        )
    ).add_to(map_obj)
    
    # Add legend if requested
    if show_legend:
        add_color_legend(map_obj, metric, bins, colors)
    
    logger.info(f"Added {len(features)} hex features to map")
    
    return map_obj


def add_color_legend(map_obj: folium.Map, metric: str, bins: List[float], colors: List[str]) -> folium.Map:
    """
    Add a color legend to the map.
    
    Args:
        map_obj: Folium Map object
        metric: Metric name for legend title
        bins: Value bins for legend
        colors: Colors corresponding to bins
        
    Returns:
        Updated Folium Map object
    """
    # Create legend HTML
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{metric.title()} Legend</b></p>
    '''
    
    for i in range(len(colors)):
        if i < len(bins) - 1:
            legend_html += f'''
            <p><i style="background:{colors[i]}; width:20px; height:20px; 
                         display:inline-block; border:1px solid black; margin-right:5px"></i>
               {bins[i]:.2f} - {bins[i+1]:.2f}</p>
            '''
        else:
            legend_html += f'''
            <p><i style="background:{colors[i]}; width:20px; height:20px; 
                         display:inline-block; border:1px solid black; margin-right:5px"></i>
               {bins[i]:.2f}+</p>
            '''
    
    legend_html += '</div>'
    
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    return map_obj


def add_markers(
    map_obj: folium.Map,
    hexes: List[str],
    df: pd.DataFrame,
    color: str = 'red',
    size: int = 8,
    popup_text: str = None
) -> folium.Map:
    """
    Add markers for selected hexes.
    
    Args:
        map_obj: Folium Map object
        hexes: List of hex IDs to mark
        df: DataFrame with hex coordinates
        color: Marker color
        size: Marker size
        popup_text: Custom popup text
        
    Returns:
        Updated Folium Map object
    """
    logger.info(f"Adding {len(hexes)} markers")
    
    # Create hex lookup
    hex_lookup = df.set_index('hex')
    
    for hex_id in hexes:
        if hex_id not in hex_lookup.index:
            continue
        
        row = hex_lookup.loc[hex_id]
        
        # Create popup text
        if popup_text is None:
            popup_text = f"""
            <b>Staging Location</b><br>
            Hex: {hex_id}<br>
            Devices: {int(row['n_devices'])}<br>
            Demand: {row.get('demand', 0):.3f}
            """
        
        # Add marker
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=size,
            popup=folium.Popup(popup_text, max_width=200),
            color='black',
            weight=2,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(map_obj)
    
    return map_obj


def add_coverage_layer(
    map_obj: folium.Map,
    selected_hexes: List[str],
    df: pd.DataFrame,
    R: int = 1,
    color: str = 'lightblue',
    opacity: float = 0.3
) -> folium.Map:
    """
    Add coverage visualization for selected staging hexes.
    
    Args:
        map_obj: Folium Map object
        selected_hexes: List of selected hex IDs
        df: DataFrame with hex features
        R: Coverage radius
        color: Coverage color
        opacity: Coverage opacity
        
    Returns:
        Updated Folium Map object
    """
    from .features import compute_hex_neighbors
    
    logger.info(f"Adding coverage layer for {len(selected_hexes)} hexes with R={R}")
    
    # Get all covered hexes
    covered_hexes = set()
    for hex_id in selected_hexes:
        neighbors = compute_hex_neighbors(hex_id, R)
        covered_hexes.update(neighbors)
    
    # Filter to hexes in our dataset
    all_hexes = set(df['hex'].tolist())
    covered_hexes = covered_hexes.intersection(all_hexes)
    
    # Create coverage features
    features = []
    for hex_id in covered_hexes:
        if hex_id not in df.set_index('hex').index:
            continue
        
        row = df[df['hex'] == hex_id].iloc[0]
        polygon = row['polygon']
        # Check if polygon is valid
        try:
            if (polygon is None or 
                not hasattr(polygon, '__len__') or 
                len(polygon) < 3):
                continue
        except (ValueError, TypeError):
            continue
        
        # Convert polygon to GeoJSON format
        polygon_coords = []
        if hasattr(polygon, 'tolist'):
            # Convert numpy array to list
            polygon_list = polygon.tolist()
        else:
            polygon_list = list(polygon)
        
        # Convert to [lng, lat] format
        for coord in polygon_list:
            if len(coord) >= 2:
                polygon_coords.append([float(coord[1]), float(coord[0])])  # [lng, lat]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            },
            "properties": {
                "hex": hex_id,
                "covered": True
            }
        }
        features.append(feature)
    
    # Add coverage layer
    if features:
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': color,
                'color': 'blue',
                'weight': 1,
                'fillOpacity': opacity,
                'opacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['hex'],
                aliases=['Covered Hex'],
                localize=True
            )
        ).add_to(map_obj)
    
    logger.info(f"Added coverage for {len(features)} hexes")
    
    return map_obj
