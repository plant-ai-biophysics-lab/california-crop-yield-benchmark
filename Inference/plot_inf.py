import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from shapely.geometry import mapping
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from datetime import datetime
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt


CRS_TARGET = "EPSG:32610"

def convert_geometry_crs_versions(geometry):
    """
    Takes a single Shapely geometry and returns two GeoDataFrames:
    one in EPSG:3310 and one in EPSG:32610.

    Parameters:
        geometry (shapely.geometry): Input geometry, no assumed CRS

    Returns:
        Tuple[GeoDataFrame, GeoDataFrame]: (gdf_3310, gdf_32610)
    """
    # Start from a GeoDataFrame with no CRS
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:3310")  # update if original is different

    # Reproject
    gdf_3310 = gdf.to_crs("EPSG:3310")
    area = gdf_3310.iloc[0]['geometry'].area
    area_acres = area / 4046.85642

    gdf_32610 = gdf.to_crs("EPSG:32610")


    return area_acres, gdf_32610.iloc[0]['geometry']

def extract_landsat(geometry):
    """
    Fully reproject Landsat raster to EPSG:32610 and crop to geometry.

    Parameters:
        geometry: Shapely geometry in EPSG:32610

    Returns:
        np.ndarray: Cropped Landsat image (6 bands)
    """
    src_path = '/data2/hkaman/Data/YieldBenchmark/counties/Merced/Raw/Landsat/2022/Merced_LT_2022_06.tif'

    with rasterio.open(src_path) as src:
        if src.crs.to_string() != CRS_TARGET:
            transform, width, height = calculate_default_transform(
                src.crs, CRS_TARGET, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': CRS_TARGET,
                'transform': transform,
                'width': width,
                'height': height
            })

            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=CRS_TARGET,
                            resampling=Resampling.nearest)

                    # Now clip the reprojected image to the geometry
                    clipped, _ = mask(dst, [mapping(geometry)], crop=True)
        else:
            clipped, _ = mask(src, [mapping(geometry)], crop=True)

    return np.nan_to_num(clipped)

def extract_et(geometry):
    """
    Fully reproject ET raster to EPSG:32610 and crop to geometry.

    Parameters:
        geometry: Shapely geometry in EPSG:32610

    Returns:
        np.ndarray: Cropped ET raster (1 band)
    """
    src_path = '/data2/hkaman/Data/YieldBenchmark/counties/Merced/Raw/ET/2022/Merced_OpenET_2022_06.tif'

    with rasterio.open(src_path) as src:
        if src.crs.to_string() != CRS_TARGET:
            transform, width, height = calculate_default_transform(
                src.crs, CRS_TARGET, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': CRS_TARGET,
                'transform': transform,
                'width': width,
                'height': height,
                'count': 1
            })

            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),  # band 1
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=CRS_TARGET,
                        resampling=Resampling.nearest
                    )
                    clipped, _ = mask(dst, [mapping(geometry)], crop=True)
        else:
            clipped, _ = mask(src, [mapping(geometry)], crop=True, indexes=1)

    return np.nan_to_num(clipped)

def extract_soil(geometry):
    path = '/data2/hkaman/Data/YieldBenchmark/counties/Merced/Raw/Soil/soil_attributes_Merced.nc'
    ds = xr.open_dataset(path)
    ds = ds.rio.write_crs(CRS_TARGET, inplace=False)
    ds = ds.rio.reproject(CRS_TARGET)

    soil_data = {}
    for var in ds.data_vars:
        clipped = ds[var].rio.clip([geometry], all_touched=True)
        soil_data[var] = clipped.values
    return soil_data

def extract_landsat_ndvi(geometry):
    """
    Reproject Landsat to EPSG:32610, crop to geometry, and return NDVI.

    Parameters:
        geometry: Shapely geometry in EPSG:32610

    Returns:
        np.ndarray: NDVI array of shape (1, H, W)
    """
    src_path = '/data2/hkaman/Data/YieldBenchmark/counties/Merced/Raw/Landsat/2022/Merced_LT_2022_06.tif'

    with rasterio.open(src_path) as src:
        if src.crs.to_string() != CRS_TARGET:
            transform, width, height = calculate_default_transform(
                src.crs, CRS_TARGET, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': CRS_TARGET,
                'transform': transform,
                'width': width,
                'height': height,
                'count': 2  # Only Red and NIR bands
            })

            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as dst:
                    # Reproject Red (band 1)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=CRS_TARGET,
                        resampling=Resampling.nearest
                    )
                    # Reproject NIR (band 4)
                    reproject(
                        source=rasterio.band(src, 4),
                        destination=rasterio.band(dst, 2),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=CRS_TARGET,
                        resampling=Resampling.nearest
                    )

                    # Mask cropped Red and NIR
                    cropped, _ = mask(dst, [mapping(geometry)], crop=True, indexes=[1, 2])
        else:
            # No reprojection needed
            cropped, _ = mask(src, [mapping(geometry)], crop=True, indexes=[1, 4])

    # Extract Red and NIR
    red = cropped[0].astype(np.float32)
    nir = cropped[1].astype(np.float32)

    # Compute NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)  # prevent divide-by-zero
    ndvi = np.expand_dims(ndvi, axis=0)     # shape: (1, H, W)

    return np.clip(ndvi, -1, 1)  # constrain NDVI range

def plot_ndvi_et_yield(ndvi, et, yield_map, titles=None, cmaps=None):
    """
    Plots NDVI, ET, and Yield in a 1x3 subplot with fixed value ranges.

    Parameters:
        ndvi (np.ndarray): NDVI array (H, W) or (1, H, W)
        et (np.ndarray): ET array (H, W) or (1, H, W)
        yield_map (np.ndarray): Disaggregated yield (H, W)
        titles (list of str): Custom titles for the subplots
        cmaps (list of str): Custom colormaps for each plot
    """

    # Remove singleton dimensions
    ndvi = np.squeeze(ndvi)
    et = np.squeeze(et)
    yield_map = np.squeeze(yield_map)

    # Set defaults
    if titles is None:
        titles = ["NDVI", "ET", "Yield (tons)"]
    if cmaps is None:
        cmaps = ["YlGn", "Blues", "OrRd"]

    data = [ndvi, et, yield_map]
    vmin = [0, 0, 0]
    vmax = [0.4, 100, 5]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, img, title, cmap, vmin_i, vmax_i in zip(axes, data, titles, cmaps, vmin, vmax):
        im = ax.imshow(img, cmap=cmap, vmin=vmin_i, vmax=vmax_i)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.show()

def disaggregate_yield_to_pixels(geometry, predicted_county_yield = None):
    """
    Distributes total yield across pixels using NDVI and ET as weights.

    Parameters:
        total_yield (float): Total yield for the entire polygon (e.g., in tons)
        ndvi (np.ndarray): NDVI array (H, W) or (1, H, W)
        et (np.ndarray): ET array (H, W) or (1, H, W)

    Returns:
        np.ndarray: Pixel-level yield map (H, W) with sum ≈ total_yield
    """
    area_acres, geometry = convert_geometry_crs_versions(geometry)
    total_yield = area_acres * predicted_county_yield if predicted_county_yield is not None else 1.0  # Default to 1 ton if not provided


    ndvi = extract_landsat_ndvi(geometry)  # shape: (1, H, W)
    et = extract_et(geometry)              # shape: (1, H, W)
    # Squeeze singleton dimensions
    ndvi = np.squeeze(ndvi)
    et = np.squeeze(et)

    # Ensure matching shapes
    assert ndvi.shape == et.shape, "NDVI and ET must have the same shape."

    # Compute weight per pixel
    weights = ndvi * (et/100)
    weights = np.clip(weights, a_min=0, a_max=None)  # Remove negative weights

    total_weight = weights.sum()

    if total_weight == 0:
        raise ValueError("Total weight is zero. Cannot disaggregate yield.")

    # Compute yield per pixel
    yield_map = (weights / total_weight) * total_yield

    return ndvi, et, yield_map  # shape: (H, W)

def plot_yield_map(yield_map, title="Disaggregated Yield (tons)", cmap="YlGn"):
    """
    Plots the yield per pixel as a heatmap.

    Parameters:
        yield_map (np.ndarray): Array of shape (H, W)
        title (str): Plot title
        cmap (str): Matplotlib colormap (default: 'YlGn')
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(yield_map, cmap=cmap)
    plt.colorbar(label="Yield (tons)")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()