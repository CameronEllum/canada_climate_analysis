# Climate Analysis

This application downloads daily climate data from MSC GeoMet and generates HTML reports featuring temperature and precipitation anomalies, trendlines, and station maps. It uses SQLite for structured data caching and Plotly for charts.

## Example

An example report is at https://cameronellum.github.io/canada_climate_analysis/reports/climate_report_monthly_1925-2026_calgary.html.

<img width="1043" height="2283" alt="image" src="https://github.com/user-attachments/assets/cdb4e30d-e6f7-4ce0-bcc2-a77b82c63e1f" />


## Prerequisites

- Python 3.14 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management

## Getting Started

1. **Clone the repository** (if you haven't already).
2. **Install dependencies** and create a virtual environment using `uv`:
   ```bash
   uv sync
   ```

## Usage

Run the analysis using `uv run main.py`. The tool will automatically geocode the location name, find nearby stations, and fetch data.

Data downloaded from MSC GeoMet is cached in a SQLite database for future use. HTTP requests can be also cached using the `requests-cache` library. This is only intended for development purposes.

### Examples

**Basic report for Calgary:**
```bash
uv run main.py --location "Calgary"
```

**Report with trendlines and standard deviation shading:**
```bash
uv run main.py --location "Toronto" --trend --std-dev
```

**Custom date range and search radius:**
```bash
uv run main.py --location "Vancouver" --start-year 1980 --end-year 2023 --radius 50
```
**Analyze maximum daily temperatures:**
```bash
uv run main.py --location "Edmonton" --max
```

**Analyze minimum daily temperatures:**
```bash
uv run main.py --location "Winnipeg" --min
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--location` | (Required) Location name to analyze. | N/A |
| `--radius` | Search radius for climate stations in km. | `100.0` |
| `--start-year` | Start year for data range. | `1900` |
| `--end-year` | End year for data range. | Current year |
| `--trend` | Include trendlines in the charts. | `True` |
| `--median` | Include median in the temperature chart. | `False` |
| `--show-anomaly` | Show anomaly heatmap coloring. | `True` |
| `--max` | Analyze maximum daily temperatures instead of mean. | `False` |
| `--min` | Analyze minimum daily temperatures instead of mean. | `False` |



## Data Sources

- Climate data provided by **MSC GeoMet** (Environment and Climate Change Canada).
- Geocoding provided by **Nominatim** (OpenStreetMap).
