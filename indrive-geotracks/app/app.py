"""
inDrive Geotracks Dashboard

A production-ready Streamlit dashboard for analyzing geospatial ride-hailing data
using H3 hexagonal indexing and advanced analytics.

This is the main application entry point that provides navigation
to different analysis pages.
"""

import streamlit as st
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="inDrive Geotracks Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 inDrive Geotracks Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Check if artifacts exist
    artifacts_dir = Path("artifacts")
    hex_files_exist = (artifacts_dir / "hex_res8.parquet").exists() and (artifacts_dir / "hex_res7.parquet").exists()
    
    if not hex_files_exist:
        st.error("⚠️ Precomputed hex features not found!")
        st.markdown("""
        Please run the preprocessing script first:
        ```bash
        python scripts/precompute_hex_features.py --input data/geo_locations_astana_hackathon --out artifacts --res 8 7
        ```
        """)
        return
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗺️ Heatmap Analysis", use_container_width=True):
            st.switch_page("pages/1_Heatmap.py")
    
    with col2:
        if st.button("📍 Driver Allocation", use_container_width=True):
            st.switch_page("pages/2_Allocation.py")
    
    with col3:
        if st.button("⚠️ Safety Analysis", use_container_width=True):
            st.switch_page("pages/3_Safety.py")
    
    st.markdown("---")
    
    # Main content
    st.markdown("""
    ## Welcome to the inDrive Geotracks Dashboard
    
    This dashboard provides comprehensive analysis of geospatial ride-hailing data
    using advanced H3 hexagonal indexing and machine learning techniques.
    """)
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Analysis Features
        
        **Heatmap Analysis**
        - Interactive H3 hexagon visualization
        - Demand scoring and normalization
        - Multiple metric overlays
        - Real-time filtering and exploration
        
        **Driver Allocation**
        - Greedy coverage optimization
        - Baseline comparison
        - Coverage metrics and KPIs
        - Interactive staging location selection
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Technical Features
        
        **Safety Analysis**
        - Anomaly detection using z-scores
        - Isolation Forest outlier detection
        - Neighborhood statistical analysis
        - Risk assessment and flagging
        
        **Data Processing**
        - Memory-efficient chunked processing
        - H3 resolution optimization
        - Real-time caching and performance
        - Windows-compatible implementation
        """)
    
    # Privacy notice
    st.markdown("""
    <div class="info-box">
    <h4>🔒 Privacy & Data Protection</h4>
    <p>This dashboard processes only aggregated, anonymized statistics. No personally 
    identifiable information (PII) is stored or displayed. All analysis is performed 
    on hex-level aggregates with minimum device thresholds to ensure privacy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    if hex_files_exist:
        try:
            import pandas as pd
            
            # Load quick stats
            df_res8 = pd.read_parquet("artifacts/hex_res8.parquet")
            df_res7 = pd.read_parquet("artifacts/hex_res7.parquet")
            
            st.markdown("### 📈 Quick Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Resolution 8 Hexes",
                    f"{len(df_res8):,}",
                    help="High-resolution hexagons for detailed analysis"
                )
            
            with col2:
                st.metric(
                    "Resolution 7 Hexes", 
                    f"{len(df_res7):,}",
                    help="Medium-resolution hexagons for broader coverage"
                )
            
            with col3:
                total_devices = df_res8['n_devices'].sum()
                st.metric(
                    "Total Devices",
                    f"{total_devices:,}",
                    help="Unique devices across all hexes"
                )
            
            with col4:
                avg_demand = df_res8['demand'].mean()
                st.metric(
                    "Avg Demand Score",
                    f"{avg_demand:.3f}",
                    help="Average normalized demand score"
                )
                
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")
    
    # Getting started
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Heatmap Analysis**: Start by exploring demand patterns across Astana
    2. **Driver Allocation**: Optimize staging locations for maximum coverage
    3. **Safety Analysis**: Identify potential risk areas and anomalies
    
    Use the navigation buttons above or the sidebar to access each analysis module.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    inDrive Geotracks Dashboard v1.0 | Built for Astana Hackathon 2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
