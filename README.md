# Climate Analysis

This application downloads daily climate data from MSC GeoMet and generates HTML reports featuring temperature and precipitation anomalies, trendlines, and station maps. It uses SQLite for structured data caching and HTTP caching to minimize redundant API requests.

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

Data downloaded from MSC GeoMet is cached in a SQLite database for future use. HTTP requests are also cached using the `requests-cache` library. This can probably be removed in the future; it was just to be a good citizen while developing.

### Examples

**Basic report for Calgary:**
```bash
uv run main.py --location "Calgary"
```

**Report with trendlines and standard deviation shading:**
```bash
uv run main.py --location "Toronto" --trend --shade-deviation
```

**Custom date range and search radius:**
```bash
uv run main.py --location "Vancouver" --start-year 1980 --end-year 2023 --radius 50
```
**Analyze maximum daily temperatures:**
```bash
uv run main.py --location "Edmonton" --max
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--location` | (Required) Location name to analyze. | N/A |
| `--radius` | Search radius for climate stations in km. | `100.0` |
| `--start-year` | Start year for data range. | `1900` |
| `--end-year` | End year for data range. | Current year |
| `--trend` | Include trendlines in the charts. | `True` |
| `--shade-deviation` | Shade standard deviation in the charts. | `False` |
| `--show-anomaly` | Show anomaly heatmap coloring. | `True` |
| `--max` | Analyze maximum daily temperatures instead of mean. | `False` |

## Example Output

*The example output will be included here later.*

## Data Sources

- Climate data provided by **MSC GeoMet** (Environment and Climate Change Canada).
- Geocoding provided by **Nominatim** (OpenStreetMap).
