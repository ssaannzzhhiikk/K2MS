"""
Heatmap Analysis Page

Interactive visualization of demand patterns and other metrics
across H3 hexagons in Astana.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from indrive.viz import make_map, add_hex_layer
from indrive.demand import get_demand_quantiles

# Configure page
st.set_page_config(
    page_title="Heatmap Analysis - inDrive Geotracks",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_hex_data(resolution: int) -> pd.DataFrame:
    """Load hex data for specified resolution."""
    try:
        df = pd.read_parquet(f"artifacts/hex_res{resolution}.parquet")
        return df
    except FileNotFoundError:
        st.error(f"Hex data for resolution {resolution} not found. Please run preprocessing first.")
        return pd.DataFrame()

def main():
    """Main heatmap analysis function."""
    
    st.title("🗺️ Heatmap Analysis")
    st.markdown("Interactive visualization of demand patterns and metrics across Astana")
    
    # Sidebar controls
    st.sidebar.header("Map Controls")
    
    # Resolution selection
    resolution = st.sidebar.selectbox(
        "H3 Resolution",
        options=[8, 7],
        index=0,
        help="Higher resolution = more detailed hexagons"
    )
    
    # Load data
    with st.spinner(f"Loading hex data for resolution {resolution}..."):
        df = load_hex_data(resolution)
    
    if df.empty:
        return
    
    # Metric selection
    metric_options = {
        'demand': 'Demand Score',
        'n_devices': 'Number of Devices',
        'n_points': 'Number of Points',
        'stop_share': 'Stop Share',
        'heading_strength': 'Heading Strength',
        'speed_p90': 'Speed P90'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Color by Metric",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x],
        index=0
    )
    
    # Filtering controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    min_devices = st.sidebar.slider(
        "Minimum Devices",
        min_value=1,
        max_value=int(df['n_devices'].max()),
        value=5,
        help="Hide hexes with fewer than this many devices"
    )
    
    # Color scheme selection
    color_schemes = {
        'YlOrRd': 'Yellow-Orange-Red',
        'Blues': 'Blue Scale',
        'Greens': 'Green Scale', 
        'Purples': 'Purple Scale',
        'Reds': 'Red Scale'
    }
    
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        options=list(color_schemes.keys()),
        format_func=lambda x: color_schemes[x],
        index=0
    )
    
    # Data statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Statistics")
    
    filtered_df = df[df['n_devices'] >= min_devices]
    
    st.sidebar.metric("Total Hexes", f"{len(df):,}")
    st.sidebar.metric("Filtered Hexes", f"{len(filtered_df):,}")
    st.sidebar.metric("Total Devices", f"{df['n_devices'].sum():,}")
    st.sidebar.metric("Avg Demand", f"{df['demand'].mean():.3f}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="metric-header">Interactive Map</p>', unsafe_allow_html=True)
        
        # Create map
        with st.spinner("Generating map..."):
            m = make_map()
            m = add_hex_layer(
                m, 
                filtered_df, 
                metric=selected_metric,
                min_devices=min_devices,
                show_legend=True
            )
            
            # Display map
            st.components.v1.html(m._repr_html_(), height=600)
    
    with col2:
        st.markdown('<p class="metric-header">Metric Details</p>', unsafe_allow_html=True)
        
        # Metric statistics
        metric_data = filtered_df[selected_metric]
        
        st.metric("Min Value", f"{metric_data.min():.3f}")
        st.metric("Max Value", f"{metric_data.max():.3f}")
        st.metric("Mean Value", f"{metric_data.mean():.3f}")
        st.metric("Median Value", f"{metric_data.median():.3f}")
        
        # Quantile information
        quantiles = get_demand_quantiles(filtered_df, [0.1, 0.25, 0.5, 0.75, 0.9])
        
        st.markdown("**Quantiles:**")
        for q_name, q_value in quantiles.items():
            st.text(f"{q_name}: {q_value:.3f}")
        
        # Top hexes
        st.markdown("**Top 5 Hexes:**")
        top_hexes = filtered_df.nlargest(5, selected_metric)[['hex', selected_metric, 'n_devices']]
        st.dataframe(top_hexes, use_container_width=True)
    
    # Detailed analysis
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    # Metric distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{metric_options[selected_metric]} Distribution")
        st.bar_chart(filtered_df[selected_metric].value_counts().head(20))
    
    with col2:
        st.subheader("Correlation Matrix")
        numeric_cols = ['n_devices', 'n_points', 'stop_share', 'speed_p90', 'heading_strength', 'demand']
        corr_matrix = filtered_df[numeric_cols].corr()
        st.dataframe(corr_matrix, use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"heatmap_data_res{resolution}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Top Hexes"):
            top_100 = filtered_df.nlargest(100, selected_metric)
            csv = top_100.to_csv(index=False)
            st.download_button(
                label="Download Top 100",
                data=csv,
                file_name=f"top_hexes_{selected_metric}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Export Statistics"):
            stats = {
                'metric': selected_metric,
                'resolution': resolution,
                'min_devices': min_devices,
                'total_hexes': len(df),
                'filtered_hexes': len(filtered_df),
                'min_value': float(metric_data.min()),
                'max_value': float(metric_data.max()),
                'mean_value': float(metric_data.mean()),
                'median_value': float(metric_data.median())
            }
            
            import json
            json_str = json.dumps(stats, indent=2)
            st.download_button(
                label="Download Stats",
                data=json_str,
                file_name=f"heatmap_stats_{selected_metric}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
