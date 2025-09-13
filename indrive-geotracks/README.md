# inDrive Geotracks Dashboard

A production-ready Streamlit dashboard for analyzing geospatial ride-hailing data using H3 hexagonal indexing and advanced analytics.

## 🚀 Features

- **Interactive Heatmap Analysis**: Visualize demand patterns across H3 hexagons
- **Driver Allocation Optimization**: Greedy algorithms for optimal staging locations
- **Safety Analysis**: Anomaly detection using statistical and ML methods
- **Real-time Performance**: Memory-efficient chunked processing with caching
- **Windows Compatible**: No external API tokens or heavy GIS dependencies

## 📋 Requirements

- Python 3.11
- Windows 10/11
- 8GB+ RAM (for processing 106MB dataset)
- No Mapbox tokens required

## 🛠️ Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd indrive-geotracks

# Create virtual environment
py -3.11 -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your dataset in the `data/` directory:

```bash
# Copy your dataset (no .csv extension needed)
copy "path\to\geo_locations_astana_hackathon" data\geo_locations_astana_hackathon
```

### 3. Precompute Hex Features

```bash
# Process the dataset and create hex aggregations
python scripts/precompute_hex_features.py --input data/geo_locations_astana_hackathon --out artifacts --res 8 7
```

This will create:
- `artifacts/hex_res8.parquet` - High-resolution hexagons
- `artifacts/hex_res7.parquet` - Medium-resolution hexagons

### 4. Launch Dashboard

```bash
# Start the Streamlit application
python -m streamlit run app/app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Usage

### Heatmap Analysis
- Select H3 resolution (7 or 8)
- Choose metric to visualize (demand, devices, speed, etc.)
- Filter by minimum devices
- Explore interactive map with tooltips

### Driver Allocation
- Set number of staging locations (k)
- Choose coverage radius (R)
- Compare baseline vs greedy strategies
- View coverage metrics and uplift

### Safety Analysis
- Configure z-score threshold for speed anomalies
- Enable/disable Isolation Forest detection
- Visualize anomalous patterns on map
- Export detailed anomaly reports

## 🏗️ Architecture

```
indrive-geotracks/
├── app/
│   ├── app.py                 # Main dashboard entry point
│   └── pages/
│       ├── 1_Heatmap.py      # Heatmap visualization
│       ├── 2_Allocation.py   # Driver allocation optimization
│       └── 3_Safety.py       # Anomaly detection
├── indrive/
│   ├── __init__.py
│   ├── io.py                  # Data loading utilities
│   ├── features.py            # H3 aggregation functions
│   ├── demand.py              # Demand scoring algorithms
│   ├── allocate.py            # Coverage optimization
│   ├── anomalies.py           # Anomaly detection methods
│   └── viz.py                 # Folium visualization helpers
├── scripts/
│   └── precompute_hex_features.py  # Data preprocessing
├── data/
│   └── .gitignore            # Ignore large data files
├── artifacts/
│   └── .gitkeep              # Precomputed hex features
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### H3 Resolution
- **Resolution 8**: ~0.74 km² hexagons (detailed analysis)
- **Resolution 7**: ~5.2 km² hexagons (broader coverage)

### Demand Scoring
The demand score combines three normalized components:
- **Devices (50%)**: Number of unique devices
- **Points (30%)**: Total data points
- **Stop Share (20%)**: Fraction of low-speed points

### Anomaly Detection
- **Z-Score Method**: Compares hex speed to neighborhood average
- **Isolation Forest**: ML-based outlier detection
- **Combined Flags**: Merges both detection methods

## 📈 Performance

- **Memory Efficient**: Chunked processing (200k rows/chunk)
- **Cached Results**: Streamlit caching for fast navigation
- **CPU Optimized**: No GPU dependencies
- **Windows Friendly**: No GDAL or complex GIS libraries

## 🔒 Privacy & Ethics

- **Anonymized Data**: Only aggregated statistics, no PII
- **Minimum Thresholds**: Hide hexes with <5 devices
- **Local Processing**: No external API calls
- **Transparent Methods**: Open-source algorithms

## 🚀 Extensions

### Adding Timestamp Support
When timestamped data becomes available:

1. Add time-based filtering to `io.py`
2. Create temporal aggregation functions
3. Add time-of-day analysis to heatmap
4. Implement origin-destination flow analysis

### Custom Metrics
To add new demand metrics:

1. Extend `demand.py` with new scoring functions
2. Update `viz.py` color scales
3. Add UI controls in Streamlit pages

### Additional Anomaly Methods
To add new detection algorithms:

1. Implement in `anomalies.py`
2. Add parameters to Safety page
3. Update visualization logic

## 🐛 Troubleshooting

### Common Issues

**"Hex data not found"**
- Run preprocessing script first
- Check file paths in `artifacts/` directory

**Memory errors during processing**
- Reduce chunk size in preprocessing
- Use lower H3 resolution
- Close other applications

**Map not displaying**
- Check internet connection (for tile loading)
- Try different browser
- Clear Streamlit cache

### Performance Tips

- Use resolution 7 for faster processing
- Increase minimum devices filter
- Close unused browser tabs
- Run on SSD storage

## 📝 License

Built for inDrive Astana Hackathon 2024

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

For technical issues or questions:
- Check troubleshooting section
- Review error logs in terminal
- Create issue with detailed description
