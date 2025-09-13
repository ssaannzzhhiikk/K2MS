"""
Safety Analysis Page

Anomaly detection and risk assessment for geospatial data
using statistical and machine learning methods.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from indrive.viz import make_map, add_hex_layer, add_markers
from indrive.anomalies import neighbor_stats, zscore_speed, isolation_forest_flags, combined_anomaly_flags, get_anomaly_summary

# Configure page
st.set_page_config(
    page_title="Safety Analysis - inDrive Geotracks",
    page_icon="⚠️",
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
    .anomaly-card {
        background-color: #fff2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 0.5rem 0;
    }
    .normal-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffa726;
        margin: 0.5rem 0;
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

@st.cache_data
def compute_anomalies(df: pd.DataFrame, z_threshold: float, use_isolation: bool) -> pd.DataFrame:
    """Compute anomaly flags for the dataset."""
    # Compute neighborhood statistics
    df_with_neighbors = neighbor_stats(df, df['hex'].iloc[0][:2])  # Extract resolution from hex
    
    # Compute z-score anomalies
    df_with_zscore = zscore_speed(df_with_neighbors, z_threshold)
    
    # Compute isolation forest anomalies if requested
    if use_isolation:
        df_with_isolation = isolation_forest_flags(df_with_zscore)
    else:
        df_with_isolation = df_with_zscore.copy()
        df_with_isolation['isolation_anomaly'] = False
        df_with_isolation['anomaly_score'] = 0.0
    
    # Combine all anomaly flags
    df_final = combined_anomaly_flags(df_with_isolation)
    
    return df_final

def main():
    """Main safety analysis function."""
    
    st.title("⚠️ Safety Analysis")
    st.markdown("Anomaly detection and risk assessment for geospatial patterns")
    
    # Sidebar controls
    st.sidebar.header("Detection Parameters")
    
    # Resolution selection
    resolution = st.sidebar.selectbox(
        "H3 Resolution",
        options=[8, 7],
        index=0,
        help="Higher resolution = more detailed analysis"
    )
    
    # Load data
    with st.spinner(f"Loading hex data for resolution {resolution}..."):
        df = load_hex_data(resolution)
    
    if df.empty:
        return
    
    # Anomaly detection parameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Z-Score Detection")
    
    z_threshold = st.sidebar.slider(
        "Z-Score Threshold",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Higher values = fewer anomalies detected"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Isolation Forest")
    
    use_isolation = st.sidebar.checkbox(
        "Enable Isolation Forest",
        value=False,
        help="Use machine learning for outlier detection"
    )
    
    if use_isolation:
        contamination = st.sidebar.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Expected proportion of outliers"
        )
    
    # Filtering
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    min_devices = st.sidebar.slider(
        "Minimum Devices",
        min_value=1,
        max_value=int(df['n_devices'].max()),
        value=5,
        help="Minimum devices required for analysis"
    )
    
    # Compute anomalies
    with st.spinner("Computing anomaly detection..."):
        try:
            df_anomalies = compute_anomalies(df, z_threshold, use_isolation)
        except Exception as e:
            st.error(f"Anomaly detection failed: {e}")
            return
    
    # Filter data
    filtered_df = df_anomalies[df_anomalies['n_devices'] >= min_devices].copy()
    
    if len(filtered_df) == 0:
        st.error("No hexes meet the minimum device requirement!")
        return
    
    # Get anomaly summary
    summary = get_anomaly_summary(filtered_df)
    
    # Display results
    st.markdown('<p class="metric-header">Anomaly Detection Results</p>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Hexes",
            f"{summary['total_hexes']:,}",
            help="Total hexes analyzed"
        )
    
    with col2:
        st.metric(
            "Speed Anomalies",
            f"{summary['speed_anomalies']:,}",
            f"{summary['speed_anomalies_pct']:.1f}%",
            help="Hexes with unusual speed patterns"
        )
    
    with col3:
        st.metric(
            "Isolation Anomalies",
            f"{summary['isolation_anomalies']:,}",
            f"{summary['isolation_anomalies_pct']:.1f}%",
            help="Hexes flagged by Isolation Forest"
        )
    
    with col4:
        st.metric(
            "Any Anomalies",
            f"{summary['any_anomalies']:,}",
            f"{summary['any_anomalies_pct']:.1f}%",
            help="Hexes with any type of anomaly"
        )
    
    # Anomaly breakdown
    st.markdown("### 📊 Anomaly Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Methods")
        
        # Speed anomalies
        speed_anomalies = filtered_df[filtered_df['speed_anomaly']]
        if len(speed_anomalies) > 0:
            st.markdown(f"""
            <div class="anomaly-card">
            <h4>🚨 Speed Anomalies ({len(speed_anomalies):,})</h4>
            <p>Hexes with unusual speed patterns relative to neighbors</p>
            <p><strong>Z-score threshold:</strong> {z_threshold}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="normal-card">
            <h4>✅ No Speed Anomalies</h4>
            <p>All hexes have normal speed patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Isolation anomalies
        if use_isolation:
            isolation_anomalies = filtered_df[filtered_df['isolation_anomaly']]
            if len(isolation_anomalies) > 0:
                st.markdown(f"""
                <div class="anomaly-card">
                <h4>🤖 Isolation Forest Anomalies ({len(isolation_anomalies):,})</h4>
                <p>Hexes flagged as outliers by machine learning</p>
                <p><strong>Contamination rate:</strong> {contamination:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="normal-card">
                <h4>✅ No Isolation Anomalies</h4>
                <p>No outliers detected by machine learning</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Anomaly Statistics")
        
        if 'anomaly_features' in summary:
            features = summary['anomaly_features']
            st.metric("Avg Speed P90", f"{features['avg_speed_p90']:.1f}")
            st.metric("Avg Heading Strength", f"{features['avg_heading_strength']:.3f}")
            st.metric("Avg Demand", f"{features['avg_demand']:.3f}")
        
        # Z-score distribution
        if 'speed_zscore' in filtered_df.columns:
            st.subheader("Z-Score Distribution")
            z_scores = filtered_df['speed_zscore'].dropna()
            st.bar_chart(z_scores.value_counts().head(20))
    
    # Interactive map
    st.markdown("### 🗺️ Anomaly Map")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_speed_anomalies = st.checkbox("Show Speed Anomalies", value=True)
    with col2:
        show_isolation_anomalies = st.checkbox("Show Isolation Anomalies", value=True)
    with col3:
        show_normal = st.checkbox("Show Normal Hexes", value=False)
    
    # Create map
    with st.spinner("Generating anomaly map..."):
        m = make_map()
        
        # Add normal hexes
        if show_normal:
            normal_df = filtered_df[~filtered_df['any_anomaly']]
            if len(normal_df) > 0:
                m = add_hex_layer(
                    m,
                    normal_df,
                    metric='demand',
                    min_devices=min_devices,
                    show_legend=False
                )
        
        # Add speed anomalies
        if show_speed_anomalies:
            speed_anomalies = filtered_df[filtered_df['speed_anomaly']]
            if len(speed_anomalies) > 0:
                m = add_hex_layer(
                    m,
                    speed_anomalies,
                    metric='speed_p90',
                    min_devices=min_devices,
                    show_legend=False
                )
                
                # Add markers for speed anomalies
                m = add_markers(
                    m,
                    speed_anomalies['hex'].tolist(),
                    speed_anomalies,
                    color='red',
                    size=8,
                    popup_text="<b>Speed Anomaly</b><br>High z-score detected"
                )
        
        # Add isolation anomalies
        if show_isolation_anomalies and use_isolation:
            isolation_anomalies = filtered_df[filtered_df['isolation_anomaly']]
            if len(isolation_anomalies) > 0:
                m = add_markers(
                    m,
                    isolation_anomalies['hex'].tolist(),
                    isolation_anomalies,
                    color='orange',
                    size=8,
                    popup_text="<b>Isolation Anomaly</b><br>ML outlier detected"
                )
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=600)
    
    # Detailed anomaly analysis
    st.markdown("### 🔍 Detailed Analysis")
    
    # Anomaly table
    anomaly_df = filtered_df[filtered_df['any_anomaly']].copy()
    
    if len(anomaly_df) > 0:
        st.subheader(f"Anomalous Hexes ({len(anomaly_df):,})")
        
        # Select columns to display
        display_cols = ['hex', 'n_devices', 'speed_p90', 'heading_strength', 'demand', 'anomaly_reason']
        if 'speed_zscore' in anomaly_df.columns:
            display_cols.append('speed_zscore')
        if 'anomaly_score' in anomaly_df.columns:
            display_cols.append('anomaly_score')
        
        available_cols = [col for col in display_cols if col in anomaly_df.columns]
        
        st.dataframe(
            anomaly_df[available_cols].sort_values('speed_p90', ascending=False),
            use_container_width=True
        )
        
        # Anomaly patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Speed vs Demand")
            st.scatter_chart(
                anomaly_df[['speed_p90', 'demand']].rename(columns={'speed_p90': 'Speed P90', 'demand': 'Demand'})
            )
        
        with col2:
            st.subheader("Heading Strength Distribution")
            st.bar_chart(anomaly_df['heading_strength'].value_counts().head(20))
    
    else:
        st.markdown("""
        <div class="normal-card">
        <h4>✅ No Anomalies Detected</h4>
        <p>All hexes appear to have normal patterns based on current parameters.</p>
        <p>Try adjusting the detection parameters in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export All Anomalies"):
            csv = anomaly_df.to_csv(index=False)
            st.download_button(
                label="Download Anomalies",
                data=csv,
                file_name=f"anomalies_res{resolution}_z{z_threshold}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Speed Anomalies"):
            speed_anomalies = filtered_df[filtered_df['speed_anomaly']]
            csv = speed_anomalies.to_csv(index=False)
            st.download_button(
                label="Download Speed Anomalies",
                data=csv,
                file_name=f"speed_anomalies_res{resolution}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Export Summary"):
            import json
            summary_json = {
                'parameters': {
                    'resolution': resolution,
                    'z_threshold': z_threshold,
                    'use_isolation': use_isolation,
                    'min_devices': min_devices
                },
                'summary': summary
            }
            json_str = json.dumps(summary_json, indent=2)
            st.download_button(
                label="Download Summary",
                data=json_str,
                file_name=f"anomaly_summary_res{resolution}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
