"""
Driver Allocation Analysis Page

Optimization algorithms for selecting optimal staging locations
to maximize demand coverage.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from indrive.viz import make_map, add_hex_layer, add_markers, add_coverage_layer
from indrive.allocate import compare_strategies, compute_coverage_metrics

# Configure page
st.set_page_config(
    page_title="Driver Allocation - inDrive Geotracks",
    page_icon="📍",
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
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .strategy-comparison {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
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
    """Main allocation analysis function."""
    
    st.title("📍 Driver Allocation Analysis")
    st.markdown("Optimize staging locations to maximize demand coverage")
    
    # Sidebar controls
    st.sidebar.header("Allocation Parameters")
    
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
    
    # Allocation parameters
    k = st.sidebar.slider(
        "Number of Staging Locations (k)",
        min_value=1,
        max_value=min(20, len(df)),
        value=5,
        help="Number of staging locations to select"
    )
    
    R = st.sidebar.slider(
        "Coverage Radius (R)",
        min_value=0,
        max_value=2,
        value=1,
        help="K-ring distance for coverage (0 = only the hex itself)"
    )
    
    min_devices = st.sidebar.slider(
        "Minimum Devices",
        min_value=1,
        max_value=int(df['n_devices'].max()),
        value=5,
        help="Minimum devices required for a hex to be considered"
    )
    
    # Filter data
    filtered_df = df[df['n_devices'] >= min_devices].copy()
    
    if len(filtered_df) == 0:
        st.error("No hexes meet the minimum device requirement!")
        return
    
    # Run allocation algorithms
    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithm Status")
    
    with st.spinner("Running allocation algorithms..."):
        try:
            comparison = compare_strategies(filtered_df, k, R, min_devices)
        except Exception as e:
            st.error(f"Allocation failed: {e}")
            return
    
    # Display results
    st.markdown('<p class="metric-header">Allocation Results</p>', unsafe_allow_html=True)
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline_metrics = comparison['baseline']['metrics']
        st.metric(
            "Baseline Coverage",
            f"{baseline_metrics['demand_coverage_pct']:.1f}%",
            help="Coverage using top-k by demand"
        )
    
    with col2:
        greedy_metrics = comparison['greedy']['metrics']
        st.metric(
            "Greedy Coverage",
            f"{greedy_metrics['demand_coverage_pct']:.1f}%",
            help="Coverage using greedy optimization"
        )
    
    with col3:
        uplift = comparison['uplift']['demand_coverage_pct']
        st.metric(
            "Coverage Uplift",
            f"+{uplift:.1f}%",
            delta=f"{uplift:.1f}%",
            help="Improvement over baseline"
        )
    
    with col4:
        st.metric(
            "Total Demand",
            f"{greedy_metrics['total_demand']:.1f}",
            help="Total demand in dataset"
        )
    
    # Strategy comparison
    st.markdown("### 📊 Strategy Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline Strategy (Top-k by Demand)**")
        st.markdown(f"""
        - Selected hexes: {len(comparison['baseline']['hexes'])}
        - Covered hexes: {baseline_metrics['covered_hexes']:,}
        - Coverage: {baseline_metrics['coverage_pct']:.1f}%
        - Demand coverage: {baseline_metrics['demand_coverage_pct']:.1f}%
        """)
        
        # Show selected hexes
        if comparison['baseline']['hexes']:
            baseline_hexes_df = filtered_df[filtered_df['hex'].isin(comparison['baseline']['hexes'])]
            st.dataframe(
                baseline_hexes_df[['hex', 'demand', 'n_devices', 'n_points']].sort_values('demand', ascending=False),
                use_container_width=True
            )
    
    with col2:
        st.markdown("**Greedy Strategy (Coverage Optimization)**")
        st.markdown(f"""
        - Selected hexes: {len(comparison['greedy']['hexes'])}
        - Covered hexes: {greedy_metrics['covered_hexes']:,}
        - Coverage: {greedy_metrics['coverage_pct']:.1f}%
        - Demand coverage: {greedy_metrics['demand_coverage_pct']:.1f}%
        """)
        
        # Show selected hexes
        if comparison['greedy']['hexes']:
            greedy_hexes_df = filtered_df[filtered_df['hex'].isin(comparison['greedy']['hexes'])]
            st.dataframe(
                greedy_hexes_df[['hex', 'demand', 'n_devices', 'n_points']].sort_values('demand', ascending=False),
                use_container_width=True
            )
    
    # Interactive map
    st.markdown("### 🗺️ Interactive Map")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_baseline = st.checkbox("Show Baseline", value=True)
    with col2:
        show_greedy = st.checkbox("Show Greedy", value=True)
    with col3:
        show_coverage = st.checkbox("Show Coverage", value=True)
    
    # Create map
    with st.spinner("Generating map..."):
        m = make_map()
        
        # Add base hex layer
        m = add_hex_layer(
            m,
            filtered_df,
            metric='demand',
            min_devices=min_devices,
            show_legend=True
        )
        
        # Add baseline markers
        if show_baseline and comparison['baseline']['hexes']:
            m = add_markers(
                m,
                comparison['baseline']['hexes'],
                filtered_df,
                color='red',
                size=10,
                popup_text="<b>Baseline Staging</b>"
            )
        
        # Add greedy markers
        if show_greedy and comparison['greedy']['hexes']:
            m = add_markers(
                m,
                comparison['greedy']['hexes'],
                filtered_df,
                color='blue',
                size=10,
                popup_text="<b>Greedy Staging</b>"
            )
        
        # Add coverage visualization
        if show_coverage and comparison['greedy']['hexes']:
            m = add_coverage_layer(
                m,
                comparison['greedy']['hexes'],
                filtered_df,
                R=R,
                color='lightblue',
                opacity=0.3
            )
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=600)
    
    # Sensitivity analysis
    st.markdown("### 📈 Sensitivity Analysis")
    
    # Parameter sweep
    if st.button("Run Parameter Sweep"):
        with st.spinner("Running sensitivity analysis..."):
            k_values = [3, 5, 7, 10, 15]
            R_values = [0, 1, 2]
            
            results = []
            
            for k_val in k_values:
                for R_val in R_values:
                    if k_val <= len(filtered_df):
                        try:
                            comp = compare_strategies(filtered_df, k_val, R_val, min_devices)
                            results.append({
                                'k': k_val,
                                'R': R_val,
                                'baseline_coverage': comp['baseline']['metrics']['demand_coverage_pct'],
                                'greedy_coverage': comp['greedy']['metrics']['demand_coverage_pct'],
                                'uplift': comp['uplift']['demand_coverage_pct']
                            })
                        except Exception as e:
                            st.warning(f"Failed for k={k_val}, R={R_val}: {e}")
            
            if results:
                results_df = pd.DataFrame(results)
                
                # Display results
                st.dataframe(results_df, use_container_width=True)
                
                # Plot results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Coverage vs k")
                    pivot_coverage = results_df.pivot(index='k', columns='R', values='greedy_coverage')
                    st.line_chart(pivot_coverage)
                
                with col2:
                    st.subheader("Uplift vs k")
                    pivot_uplift = results_df.pivot(index='k', columns='R', values='uplift')
                    st.line_chart(pivot_uplift)
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Baseline Results"):
            baseline_hexes_df = filtered_df[filtered_df['hex'].isin(comparison['baseline']['hexes'])]
            csv = baseline_hexes_df.to_csv(index=False)
            st.download_button(
                label="Download Baseline",
                data=csv,
                file_name=f"baseline_allocation_k{k}_R{R}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Greedy Results"):
            greedy_hexes_df = filtered_df[filtered_df['hex'].isin(comparison['greedy']['hexes'])]
            csv = greedy_hexes_df.to_csv(index=False)
            st.download_button(
                label="Download Greedy",
                data=csv,
                file_name=f"greedy_allocation_k{k}_R{R}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Export Comparison"):
            import json
            comparison_json = {
                'parameters': {'k': k, 'R': R, 'min_devices': min_devices, 'resolution': resolution},
                'baseline': {
                    'hexes': comparison['baseline']['hexes'],
                    'metrics': comparison['baseline']['metrics']
                },
                'greedy': {
                    'hexes': comparison['greedy']['hexes'],
                    'metrics': comparison['greedy']['metrics']
                },
                'uplift': comparison['uplift']
            }
            json_str = json.dumps(comparison_json, indent=2)
            st.download_button(
                label="Download Comparison",
                data=json_str,
                file_name=f"allocation_comparison_k{k}_R{R}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
