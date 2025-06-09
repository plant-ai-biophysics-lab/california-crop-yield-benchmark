import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import transform
import pyproj
import os
import json
from datetime import datetime


def load_geometry_from_file(filepath):
    """Load geometry from a GeoJSON or shapefile."""
    gdf = gpd.read_file(filepath)
    return gdf.geometry[0]  # Assumes one geometry


def to_crs(geometry, from_crs="EPSG:4326", to_crs="EPSG:5070"):
    """Reproject shapely geometry from one CRS to another."""
    project = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
    return transform(project, geometry)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def log_jsonl(path, entry):
    """Append a dictionary as a JSON line to a log file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def timestamp():
    """Return current timestamp as string."""
    return datetime.now().isoformat(timespec="seconds")