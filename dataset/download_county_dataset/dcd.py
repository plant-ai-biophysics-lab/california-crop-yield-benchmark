import os
from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import wkt
import geemap
import xarray as xr
import rioxarray
import geojson
import ee
import geopandas as gpd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pynhd import NLDI
import pydaymet as daymet
import calendar
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import requests
import json
from rasterio.mask import mask


cdl_crop_legend = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    6: "Sunflower",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    33: "Safflower",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    53: "Peas",
    54: "Tomatoes",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    61: "Fallow/Idle Cropland",
    62: "Non-agricultural",
    63: "Non-agricultural",
    64: "Non-agricultural",
    65: "Non-agricultural",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Non-agricultural",
    82: "Non-agricultural",
    83: "Non-agricultural",
    84: "Non-agricultural",
    85: "Non-agricultural",
    86: "Non-agricultural",
    87: "Non-agricultural",
    88: "Non-agricultural",
    89: "Non-agricultural",
    90: "Non-agricultural",
    91: "Non-agricultural",
    93: "NLCD-sampled categories",
    # Adding NLCD-sampled categories from 94–199
    **{code: "NLCD-sampled categories" for code in range(94, 200)},
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    229: "Pumpkins",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    121: "Urban",
    141: "Water",
    0: "Other"
}

cdl_colors = {
    0: '#d3d3d3',   # Other (light gray)
    1: '#ff0000',   # Corn (red)
    2: '#ffa500',   # Cotton (orange)
    3: '#ffff00',   # Rice (yellow)
    4: '#8b4513',   # Sorghum (brown)
    6: '#ffd700',   # Sunflower (gold)
    12: '#32cd32',  # Sweet Corn (lime green)
    13: '#228b22',  # Pop or Orn Corn (forest green)
    14: '#006400',  # Mint (dark green)
    21: '#4682b4',  # Barley (steel blue)
    22: '#00bfff',  # Durum Wheat (deep sky blue)
    23: '#1e90ff',  # Spring Wheat (dodger blue)
    24: '#4169e1',  # Winter Wheat (royal blue)
    25: '#87cefa',  # Other Small Grains (light blue)
    27: '#b0e0e6',  # Rye (powder blue)
    28: '#7fffd4',  # Oats (aquamarine)
    29: '#40e0d0',  # Millet (turquoise)
    33: '#00ced1',  # Safflower (dark turquoise)
    36: '#20b2aa',  # Alfalfa (light sea green)
    37: '#5f9ea0',  # Other Hay/Non Alfalfa (cadet blue)
    41: '#4682b4',  # Sugarbeets (steel blue)
    42: '#6495ed',  # Dry Beans (cornflower blue)
    43: '#7b68ee',  # Potatoes (medium slate blue)
    44: '#6a5acd',  # Other Crops (slate blue)
    45: '#8a2be2',  # Sugarcane (blue violet)
    46: '#9370db',  # Sweet Potatoes (medium purple)
    58: '#9932cc',  # Clover/Wildflowers (dark orchid)
    59: '#ba55d3',  # Sod/Grass Seed (medium orchid)
    61: '#ff69b4',  # Fallow/Idle Cropland (hot pink)
    62: '#ff1493',  # Non-agricultural (deep pink)
    66: '#ff7f50',  # Cherries (coral)
    67: '#ff6347',  # Peaches (tomato)
    68: '#ff4500',  # Apples (orange red)
    69: '#dc143c',  # Grapes (crimson)
    71: '#b22222',  # Other Tree Crops (firebrick)
    72: '#a52a2a',  # Citrus (brown)
    74: '#800000',  # Pecans (maroon)
    75: '#808000',  # Almonds (olive)
    76: '#556b2f',  # Walnuts (dark olive green)
    77: '#6b8e23',  # Pears (olive drab)
    81: '#2e8b57',  # Non-agricultural (sea green)
    141: '#4682b4', # Water (steel blue)
    204: '#f0e68c', # Pistachios (khaki)
    205: '#bdb76b', # Triticale (dark khaki)
    206: '#9acd32', # Carrots (yellow green)
    207: '#556b2f', # Asparagus (dark olive green)
    208: '#8fbc8f', # Garlic (dark sea green)
    209: '#7fff00', # Cantaloupes (chartreuse)
    211: '#adff2f', # Olives (green yellow)
    212: '#98fb98', # Oranges (pale green)
    213: '#00ff7f', # Honeydew Melons (spring green)
    214: '#3cb371', # Broccoli (medium sea green)
    216: '#2e8b57', # Peppers (sea green)
    217: '#008b8b', # Pomegranates (dark cyan)
    218: '#00ced1', # Nectarines (dark turquoise)
    219: '#20b2aa', # Greens (light sea green)
    220: '#5f9ea0', # Plums (cadet blue)
    221: '#4682b4', # Strawberries (steel blue)
    222: '#87ceeb', # Squash (sky blue)
    223: '#6495ed', # Apricots (cornflower blue)
    224: '#7b68ee', # Vetch (medium slate blue)
    225: '#6a5acd', # Dbl Crop WinWht/Corn (slate blue)
    226: '#8a2be2', # Dbl Crop Oats/Corn (blue violet)
    227: '#9370db', # Lettuce (medium purple)
    229: '#9932cc', # Pumpkins (dark orchid)
    231: '#ba55d3', # Dbl Crop Lettuce/Cantaloupe (medium orchid)
    232: '#ff69b4', # Dbl Crop Lettuce/Cotton (hot pink)
    236: '#ff1493', # Dbl Crop WinWht/Sorghum (deep pink)
    237: '#dc143c', # Dbl Crop Barley/Corn (crimson)
    238: '#b22222', # Dbl Crop WinWht/Cotton (firebrick)
    242: '#a52a2a', # Blueberries (brown)
    243: '#800000', # Cabbage (maroon)
    244: '#808000', # Cauliflower (olive)
    245: '#556b2f', # Celery (dark olive green)
    246: '#6b8e23', # Radishes (olive drab)
    247: '#2e8b57', # Turnips (sea green)
}

class DownloadSatelliteImgEE:
    def __init__(self, 
                 year: int, 
                 start_date: str, 
                 end_date: str,
                 cloud_filter: float, 
                 satellite: str):
        self.year = year
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_filter = cloud_filter
        self.satellite = satellite

        # Load California counties
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=4326)

    def __call__(self):
        self.get_satellite_data_for_county()

    def get_satellite_data_for_county(self):
        polygon = self.dataframe.iloc[19].geometry
        aoi_geojson = geojson.Feature(geometry=mapping(polygon))
        geometry = aoi_geojson["geometry"]

        try:
            ee_geometry = self.get_flexible_geometry(geometry)
        except ValueError as e:
            print(f"Error processing geometry: {e}")
            return

        county_name = self.dataframe.iloc[19].NAME
        county_name_modified = self.county_name_modification(county_name)

        # for i in range(12):
        # start_date, end_date = self.get_monthly_dates(i)
        # print(start_date, end_date)
        images = self.get_landsat_images_by_month(geometry, ee_geometry, self.start_date, self.end_date)
        for image in images:
            if image is not None:
                self.export_image(image, county_name_modified, ee_geometry, index=0)


    def get_flexible_geometry(self, geometry):
        """
        Convert a GeoPandas geometry to an Earth Engine-compatible geometry.

        Args:
            geometry (shapely.geometry): The input geometry (Polygon, MultiPolygon, etc.).

        Returns:
            ee.Geometry: A valid Earth Engine geometry.
        """
        # Handle MultiPolygon or Polygon for Earth Engine
        if geometry["type"] == "Polygon":
            eeg = ee.Geometry.Polygon(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            # Flatten MultiPolygon to list of Polygons for Earth Engine
            polygons = [coords for coords in geometry["coordinates"]]
            eeg = ee.Geometry.MultiPolygon(polygons)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")
        return eeg
   
    def county_name_modification(self, county_name: str) -> str:
        if county_name.endswith(" County"):
            county_name = county_name[:-7]
        county_name = county_name.replace(" ", "")
        return county_name
    
    def get_year_dates(self):
        start_date = f'{self.year}-01-01'
        end_date = f'{self.year}-12-31'  
        return start_date, end_date
    
    def get_monthly_dates(self, index):
        """
        Returns the start and end dates for a specific month in the given year.

        Args:
            index (int): The month index (0 for January, 1 for February, ..., 11 for December).

        Returns:
            tuple: A tuple containing the start_date and end_date in 'YYYY-MM-DD' format.
        """
        if index < 0 or index > 11:
            raise ValueError("Index must be between 0 (January) and 11 (December).")
        
        # Convert index to 1-based month (e.g., 0 -> 1, 1 -> 2, ...)
        month = index + 1

        # Get the last day of the month
        last_day = calendar.monthrange(self.year, month)[1]

        # Format the start and end dates
        start_date = f'{self.year}-{month:02d}-01'
        end_date = f'{self.year}-{month:02d}-{last_day}'

        return start_date, end_date
    
    def get_image_size(self, image):
        """
        Calculate the approximate size of an Earth Engine image in megabytes (MB).

        Args:
            image (ee.Image): The Earth Engine image.

        Returns:
            ee.Number: The size of the image in MB.
        """
        # Get the number of pixels in the image
        num_pixels = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=image.geometry(),
            scale=30,  # Landsat resolution
            maxPixels=1e13
        ).values().get(0)  # First (and only) value

        # Get the number of bands in the image
        num_bands = image.bandNames().size()

        # Assume 16-bit for Landsat bands
        bit_depth = 16

        # Calculate size in MB
        size_mb = ee.Number(num_pixels).multiply(num_bands).multiply(bit_depth).divide(8 * 1024**2)

        return size_mb

    def has_missing_values(self, image):
        """
        Check if an image has missing (masked) values.

        Args:
            image (ee.Image): The Earth Engine image.

        Returns:
            bool: True if the image has missing values, otherwise False.
        """
        mask = image.mask()
        missing = mask.Not()  # Invert the mask to find missing pixels
        missing_count = missing.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e13
        ).getInfo()

        return any(value > 0 for value in missing_count.values())
    
    def fill_large_gaps_with_mosaic(self, collection, geometry):
        """
        Fills large gaps in Landsat images by combining multiple images in a collection.

        Args:
            collection (ee.ImageCollection): The Landsat image collection.
            geometry (ee.Geometry): The region of interest.

        Returns:
            ee.Image: A single image with reduced gaps.
        """
        # Mosaic multiple images to fill gaps
        combined_image = collection.mosaic().clip(geometry)
        return combined_image
    
    def fill_with_focal_mean(self, image):
        """
        Fills smaller gaps in an image using neighboring pixels.

        Args:
            image (ee.Image): The Landsat image.

        Returns:
            ee.Image: The image with filled gaps.
        """
        return image.unmask(None).focal_mean(radius=9, units='pixels', kernelType='square')

    def get_landsat_images_by_month(self, geometry, ee_geometry, start_date, end_date):
        """
        Fetches a Landsat image for one month, filtered by geometry, cloud cover, and size.
        Ensures the image with the lowest cloud cover and size ≥ 100 MB is selected.

        Args:
            geometry (ee.Geometry): The region of interest.

        Returns:
            ee.Image: The selected Landsat image or None if no suitable image is found.
        """
        # Define Landsat collections
        landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
        landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])

        # Merge collections and filter by geometry and date
        landsat = (landsat5.merge(landsat7).merge(landsat8)
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                # .filter(ee.Filter.lt('CLOUD_COVER', 5))
                )
        # # Combine multiple images using mosaic to fill large gaps
        # landsat = self.fill_large_gaps_with_mosaic(landsat, geometry)
        # # Fill remaining small gaps using focal_mean
        # landsat = self.fill_with_focal_mean(landsat)
        image_list = landsat.toList(landsat.size())
        # image = ee.Image(image_list.get(0))


        sorted_landsat = landsat.sort('CLOUD_COVER')
        # # best_image = None
        # image_list = sorted_landsat.toList(sorted_landsat.size())

        list_images = []
        # num_images = sorted_landsat.size().getInfo()
        # print(f"Number of images: {num_images}")
        for i in range(sorted_landsat.size().getInfo()):
            image = ee.Image(image_list.get(i))
        #     size_mb = self.get_image_size(image).getInfo()
            image = self.crop_image_to_geometry(image, ee_geometry)
            list_images.append(image)
        #     if size_mb >= 100:
        #         best_image = image
        #         # best_image = self.fill_missing_values(best_image)
        #         print(f"Selected image with cloud cover: {best_image.get('CLOUD_COVER').getInfo()} and size: {size_mb:.2f} MB")
        #         break
        #     else:
        #         print(f"Image skipped due to size: {size_mb:.2f} MB")

        # if best_image is None:
        #     print("No suitable Landsat image found for the specified month.")

        # print(f"Selected image ID: {best_image.get('system:id').getInfo()} with cloud cover: {best_image.get('CLOUD_COVER').getInfo()}")
        return list_images

    def crop_image_to_geometry(self, image, geometry):
        """
        Crop the given image to the specified geometry.

        Args:
            image (ee.Image): The Earth Engine Image object.
            geometry (ee.Geometry): The geometry to crop the image to.
        
        Returns:
            ee.Image: Cropped Earth Engine Image.
        """
        return image.clip(geometry)

    def export_image(self, image, county_name, ee_geometry, index=None):
        image = image.toFloat()

        try:
            image_id = image.get('system:id').getInfo()
            file_name_part = image_id.split('/')[-1]
        except Exception:
            file_name_part = f"image_{index}"  # Fallback identifier if system:id is missing

        file_name = f"{county_name}_{file_name_part}"  
        year_folder = str(self.year)
        folder_name = f"{county_name}_{year_folder}"

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=file_name,
            folder=folder_name,
            scale=30,
            region=ee_geometry,
            maxPixels=1e13
        )
        task.start()
        print(f"Export task {file_name} started.")
        return task
    
class DownloadCDLEE():
    def __init__(self, 
                 year: int, 
                 county_name: str):
        
        self.year = year
        self.county_name = county_name

        # Load California counties
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=4326)

    def __call__(self):

        self.get_cdl_data_for_county()

    def get_cdl_data_for_county(self):

        geometry, ee_geometry, county_name_modified = self.get_county_info(self.county_name)
        cdl_by_year = self.get_cdl_data()
        cdl_county = self.crop_cdl_to_geometry(cdl_by_year, geometry)
        self.export_image(cdl_county, county_name_modified, ee_geometry)

        # return cdl_county

    def get_county_info(self, county_name):
        """
        Processes a county name to find the corresponding county in the DataFrame,
        retrieves its geometry, and prepares it for Earth Engine operations.

        Args:
            county_name (str): The name of the county (case insensitive, without "County").

        Returns:
            tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
        """
        # Normalize the input county name to match the DataFrame format
        normalized_county_name = county_name.strip().title() + " County"

        # Find the index of the matching county
        try:
            county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
        except IndexError:
            print(f"County '{county_name}' not found in the DataFrame.")
            return

        # Retrieve the geometry of the county
        polygon = self.dataframe.iloc[county_index].geometry
        aoi_geojson = geojson.Feature(geometry=mapping(polygon))
        geometry = aoi_geojson["geometry"]

        # Prepare the geometry for Earth Engine
        try:
            ee_geometry = self.get_flexible_geometry(geometry)
        except ValueError as e:
            print(f"Error processing geometry for county '{county_name}': {e}")
            return

        # Get the original county name from the DataFrame
        county_name_df = self.dataframe.iloc[county_index]["NAME"]
        county_name_modified = self.county_name_modification(county_name_df)


        return geometry, ee_geometry, county_name_modified

    def get_dates(self):
        start_date = f'{self.year}-01-01'
        end_date = f'{self.year}-12-31'  
        return start_date, end_date
    
    def county_name_modification(self, county_name: str) -> str:
        if county_name.endswith(" County"):
            county_name = county_name[:-7]
        county_name = county_name.replace(" ", "")
        return county_name
    
    def get_flexible_geometry(self, geometry):
        """
        Convert a GeoPandas geometry to an Earth Engine-compatible geometry.

        Args:
            geometry (shapely.geometry): The input geometry (Polygon, MultiPolygon, etc.).

        Returns:
            ee.Geometry: A valid Earth Engine geometry.
        """
        # Handle MultiPolygon or Polygon for Earth Engine
        if geometry["type"] == "Polygon":
            eeg = ee.Geometry.Polygon(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            # Flatten MultiPolygon to list of Polygons for Earth Engine
            polygons = [coords for coords in geometry["coordinates"]]
            eeg = ee.Geometry.MultiPolygon(polygons)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")
        return eeg
    
    def get_cdl_data(self):
        # Load CDL data for the specified year
        start_date, end_date = self.get_dates()
        cdl = ee.ImageCollection('USDA/NASS/CDL') \
            .filterDate(start_date, end_date) \
            .first()

        if not cdl:
            print(f"No CDL data found for the year {self.year}")
            return
        else:
            return cdl
        
    def crop_cdl_to_geometry(self, image, county_geometry):

        # Clip the CDL data to the county geometry
        cdl_clipped = image.clip(county_geometry)

        return cdl_clipped
    
    def export_image(self, image, county_name, ee_geometry):

        image_id = image.get('system:id').getInfo()  
        file_name_part = image_id.split('/')[-1]  
        file_name = f"{county_name}_{file_name_part}"  
        year_folder = str(self.year)
        folder_name = f"CDL_{county_name}_{year_folder}"

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=file_name,
            folder=folder_name,
            scale=30,
            region=ee_geometry,
            maxPixels=1e13
        )
        task.start()
        print(f"Export task {file_name} started.")
        return task

class DownloadClimateEE():
    """
    Downloads DAYMET daily data for a given year and county, saving it as a NetCDF file.

    Args:
        county_name (str): Name of the county (case insensitive, without "County").
        year (int): The year for which to fetch DAYMET data.
        output_file (str): Path to save the NetCDF file.

    Returns:
        xarray.Dataset: The dataset containing daily observations for all variables.
    """

    def __init__(self, county_name: str, year: int):
        self.county_name = county_name
        self.year = year 

        # Load California counties
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=4326)

    def __call__(self):
        return self.get_climate_data_county()
    

    def get_climate_data_county(self):
        geometry, county_name_modified = self.get_county_info(self.county_name)

        dataset = self.get_daymet_data(geometry)

        self.export_image(dataset, county_name_modified)

        return dataset

    def get_county_info(self, county_name):
        """
        Processes a county name to find the corresponding county in the DataFrame,
        retrieves its geometry, and prepares it for Earth Engine operations.

        Args:
            county_name (str): The name of the county (case insensitive, without "County").

        Returns:
            tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
        """
        # Normalize the input county name to match the DataFrame format
        normalized_county_name = county_name.strip().title() + " County"

        # Find the index of the matching county
        try:
            county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
        except IndexError:
            print(f"County '{county_name}' not found in the DataFrame.")
            return

        # Retrieve the geometry of the county
        geometry = self.dataframe.iloc[county_index].geometry

        # Get the original county name from the DataFrame
        county_name_df = self.dataframe.iloc[county_index]["NAME"]
        county_name_modified = self.county_name_modification(county_name_df)


        return geometry, county_name_modified

    def get_dates(self):
        start_date = f'{self.year}-01-01'
        end_date = f'{self.year}-12-31'  
        return start_date, end_date
    
    def county_name_modification(self, county_name: str) -> str:
        if county_name.endswith(" County"):
            county_name = county_name[:-7]
        county_name = county_name.replace(" ", "")
        return county_name
    
    def get_daymet_data(self, geometry):

        start_date, end_date = self.get_dates()
        dates = (start_date, end_date)
  
        variables = ["prcp", "tmin", "tmax", "vp", "srad"]
        dataset = daymet.get_bygeom(geometry, dates, variables=variables, pet="priestley_taylor", snow=True)

        return dataset

    def export_image(self, dataset, county_name):

        root_dir = '/data2/hkaman/Data/FoundationModel/Monteray/Climate'
        year_folder = str(self.year)
        folder_name = f"DayMet_{county_name}_{year_folder}.nc"

        full_name = os.path.join(root_dir, year_folder, folder_name)
        dataset.to_netcdf(full_name)
        print(f"NetCDF file saved to {full_name}")

class DownloadOpenETEE:
    def __init__(self, county_name: str, year: int):
        """
        Initialize the class for downloading OpenET data.

        Args:
            year (int): Year for which data will be downloaded.
        """
        self.year = year
        self.county_name = county_name

        # Load California counties
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=4326)

    def download(self):
        """
        Downloads OpenET data for each month in the specified year for the given geometry.
        """
        geometry, ee_geometry, county_name_modified = self.get_county_info(self.county_name)


        for i in range(12):
            start_date, end_date = self.get_monthly_dates(i)
            print(start_date, end_date)
            image = self.get_openet_image(geometry, ee_geometry, start_date, end_date)
            if image is not None:
                self.export_image(image, county_name_modified, ee_geometry, index=i)

    def get_flexible_geometry(self, geometry):
        """
        Convert a GeoPandas geometry to an Earth Engine-compatible geometry.

        Args:
            geometry (shapely.geometry): The input geometry (Polygon, MultiPolygon, etc.).

        Returns:
            ee.Geometry: A valid Earth Engine geometry.
        """
        if geometry["type"] == "Polygon":
            return ee.Geometry.Polygon(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            polygons = [coords for coords in geometry["coordinates"]]
            return ee.Geometry.MultiPolygon(polygons)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry['type']}")

    def get_county_info(self, county_name):
        """
        Processes a county name to find the corresponding county in the DataFrame,
        retrieves its geometry, and prepares it for Earth Engine operations.

        Args:
            county_name (str): The name of the county (case insensitive, without "County").

        Returns:
            tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
        """
        # Normalize the input county name to match the DataFrame format
        normalized_county_name = county_name.strip().title() + " County"

        # Find the index of the matching county
        try:
            county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
        except IndexError:
            print(f"County '{county_name}' not found in the DataFrame.")
            return

        # Retrieve the geometry of the county
        polygon = self.dataframe.iloc[county_index].geometry
        aoi_geojson = geojson.Feature(geometry=mapping(polygon))
        geometry = aoi_geojson["geometry"]

        # Prepare the geometry for Earth Engine
        try:
            ee_geometry = self.get_flexible_geometry(geometry)
        except ValueError as e:
            print(f"Error processing geometry for county '{county_name}': {e}")
            return

        # Get the original county name from the DataFrame
        county_name_df = self.dataframe.iloc[county_index]["NAME"]
        county_name_modified = self.county_name_modification(county_name_df)


        return geometry, ee_geometry, county_name_modified
    
    def county_name_modification(self, county_name: str) -> str:
        if county_name.endswith(" County"):
            county_name = county_name[:-7]
        county_name = county_name.replace(" ", "")
        return county_name

    def get_monthly_dates(self, index):
        """
        Returns the start and end dates for a specific month in the given year.

        Args:
            index (int): The month index (0 for January, 1 for February, ..., 11 for December).

        Returns:
            tuple: A tuple containing the start_date and end_date in 'YYYY-MM-DD' format.
        """
        if index < 0 or index > 11:
            raise ValueError("Index must be between 0 (January) and 11 (December).")
        
        month = index + 1
        last_day = calendar.monthrange(self.year, month)[1]
        start_date = f'{self.year}-{month:02d}-01'
        end_date = f'{self.year}-{month:02d}-{last_day}'
        return start_date, end_date

    def get_openet_image(self, geometry, ee_geometry, start_date, end_date):
        """
        Fetches OpenET data for a specific month.

        Args:
            geometry (ee.Geometry): The region of interest.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
            ee.Image: The OpenET image for the specified month.
        """
        openet_collection = ee.ImageCollection('OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0')
        openet = (openet_collection
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  .mean()  # Reduce to a single image for the month
                  .clip(ee_geometry))  
        
        return openet

    def export_image(self, image, county_name, ee_geometry, index=None):
        """
        Exports the OpenET image to Google Drive.

        Args:
            image (ee.Image): The Earth Engine Image object.
            county_name (str): Modified county name for naming the output file.
            ee_geometry (ee.Geometry): Geometry for clipping the image.
            index (int, optional): Month index for naming the output file.
        """
        file_name = f"{county_name}_OpenET_{self.year}_{index+1:02d}"
        folder_name = f"OpenET_{county_name}_{self.year}"

        task = ee.batch.Export.image.toDrive(
            image=image.clip(ee_geometry),
            description=file_name,
            folder=folder_name,
            scale=30,
            region=ee_geometry,
            maxPixels=1e13
        )
        task.start()
        print(f"Export task {file_name} started.")

class SSURGODownloader:
    def __init__(self, county_name: str, year: int):
        self.county_name = county_name
        self.year = year 

        # Load California counties
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=4326)

        """
        Initialize the SSURGODownloader.

        Args:
            output_dir (str): Directory where the downloaded files will be stored.
        """
        
        self.base_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService.asmx"


    def get_soil_data(self):
        """
        Download SSURGO data for a given geometry.

        Args:
            geometry (dict or GeoDataFrame): GeoJSON geometry or GeoDataFrame defining the area of interest.
            file_name (str): Name of the output ZIP file.

        Returns:
            str: Path to the downloaded ZIP file.
        """
        # Convert GeoDataFrame to GeoJSON if needed
        file_name="soil_data.zip"
        geometry, county_name_modified = self.get_county_info(self.county_name)


        # if isinstance(geometry, gpd.GeoDataFrame):
        geometry_json = self.prepare_geometry(geometry)
        payload = {
            "polygon": geometry_json,
            "format": "GeoJSON"
        }
        response = requests.post(
            f"{self.base_url}/GetSoilData?polygonType=GeoJSON",
            json=payload
        )
        self.output_dir = os.path.join('/data2/hkaman/Data/FoundationModel', self.county_name, 'Soil')
        print(self.output_dir)


        if response.status_code == 200:
            zip_path = os.path.join(self.output_dir, file_name)
            with open(zip_path, "wb") as file:
                file.write(response.content)
            print(f"SSURGO data downloaded successfully: {zip_path}")
            return zip_path
        else:
            raise Exception(f"Failed to download SSURGO data: {response.status_code} - {response.text}")
    
    def prepare_geometry(self, geometry):
        if isinstance(geometry, gpd.GeoDataFrame):
            geometry = geometry.geometry.unary_union
        geometry_dict = mapping(geometry)
        geometry_json = json.dumps(geometry_dict)
        return geometry_json


    def get_county_info(self, county_name):
        """
        Processes a county name to find the corresponding county in the DataFrame,
        retrieves its geometry, and prepares it for Earth Engine operations.

        Args:
            county_name (str): The name of the county (case insensitive, without "County").

        Returns:
            tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
        """
        # Normalize the input county name to match the DataFrame format
        normalized_county_name = county_name.strip().title() + " County"

        # Find the index of the matching county
        try:
            county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
        except IndexError:
            print(f"County '{county_name}' not found in the DataFrame.")
            return

        # Retrieve the geometry of the county
        geometry = self.dataframe.iloc[county_index].geometry

        # Get the original county name from the DataFrame
        county_name_df = self.dataframe.iloc[county_index]["NAME"]
        county_name_modified = self.county_name_modification(county_name_df)


        return geometry, county_name_modified
    
    def county_name_modification(self, county_name: str) -> str:
        if county_name.endswith(" County"):
            county_name = county_name[:-7]
        county_name = county_name.replace(" ", "")
        return county_name
    # def extract_soil_data(self, zip_path):
    #     """
    #     Extract the downloaded ZIP file.

    #     Args:
    #         zip_path (str): Path to the ZIP file.

    #     Returns:
    #         str: Path to the extracted folder.
    #     """
    #     import zipfile

    #     extract_dir = os.path.splitext(zip_path)[0]
    #     with zipfile.ZipFile(zip_path, "r") as zip_ref:
    #         zip_ref.extractall(extract_dir)
    #     print(f"SSURGO data extracted to: {extract_dir}")
    #     return extract_dir


def count_tif_files(folder_path):
    """
    Count the number of .tif files within a folder.

    Args:
        folder_path (str): The path to the folder to search for .tif files.

    Returns:
        int: The number of .tif files in the folder.
    """
    # Ensure the folder path exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path {folder_path} does not exist.")
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Count .tif files
    tif_count = sum(1 for file in files if file.lower().endswith('.tif'))
    
    return tif_count

def extract_all_zip_files(folder_path):
    """
    Extract all .zip files within a folder to the same folder.

    Args:
        folder_path (str): The path to the folder containing the .zip files.

    Returns:
        None
    """
    # Ensure the folder path exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path {folder_path} does not exist.")
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through the files and extract .zip files
    for file in files:
        if file.endswith('.zip'):
            zip_path = os.path.join(folder_path, file)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to the same folder
                    zip_ref.extractall(folder_path)
                print(f"Extracted: {file}")
            except zipfile.BadZipFile:
                print(f"Error: {file} is not a valid zip file.")

def plot_landsat5_rgb(tif_file_path):
    """
    Plot an RGB image from a Landsat 5 surface reflectance .tif file, reproject to EPSG:3857, and print shapes.

    Args:
        tif_file_path (str): Path to the .tif file.
    
    Returns:
        None. Displays the RGB image.
    """
    def normalize(array):
        """
        Normalize the array to [0, 1] for better visualization after handling NaN values.
        
        Args:
            array (numpy.ndarray): Input array to normalize.
        
        Returns:
            numpy.ndarray: Normalized array.
        """
        nan_mask = np.isnan(array)
        if nan_mask.any():
            array[nan_mask] = np.nanmin(array)
        return (array - array.min()) / (array.max() - array.min())
    
    # Open the Landsat file
    with rasterio.open(tif_file_path) as src:
        # Read bands
        red_band = src.read(3)  # Red is band 3 in Landsat 5
        green_band = src.read(2)  # Green is band 2
        blue_band = src.read(1)  # Blue is band 1

        # Print original shape
        print(f"Original shape: {red_band.shape}")
        print(f"Original CRS: {src.crs}")

        # Target CRS
        target_crs = "EPSG:3857"

        # Calculate transform and new shape for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Prepare arrays for reprojected bands
        red_reprojected = np.empty((height, width), dtype=red_band.dtype)
        green_reprojected = np.empty((height, width), dtype=green_band.dtype)
        blue_reprojected = np.empty((height, width), dtype=blue_band.dtype)

        # Reproject each band
        reproject(
            source=red_band,
            destination=red_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        reproject(
            source=green_band,
            destination=green_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        reproject(
            source=blue_band,
            destination=blue_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        # Print reprojected shape
        print(f"Reprojected shape: {red_reprojected.shape}")
        print(f"Reprojected CRS: {target_crs}")

    # Normalize bands for visualization
    red = normalize(red_reprojected)
    green = normalize(green_reprojected)
    blue = normalize(blue_reprojected)

    # Stack bands into RGB
    rgb = np.stack([red, green, blue], axis=-1)

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis('off')  # Remove axis for better visualization
    plt.title("Reprojected Landsat 5 RGB Image (EPSG:3857)")
    plt.show()

def list_tif_files(folder_path):
    """
    List all `.tif` files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing `.tif` files.
    
    Returns:
        list: A list of `.tif` file names.
    """
    return sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

def read_tif_by_index(folder_path, index):
    """
    Read a `.tif` file by its index using xarray.
    
    Args:
        folder_path (str): Path to the folder containing `.tif` files.
        index (int): Index of the `.tif` file in the list.
    
    Returns:
        xarray.Dataset or xarray.DataArray: The read `.tif` file.
    """
    files = list_tif_files(folder_path)
    if index < 0 or index >= len(files):
        raise IndexError("Index out of range.")
    print(f"image name: {files[index]}")
    file_path = os.path.join(folder_path, files[index])
    return xr.open_dataset(file_path, engine="rasterio")#.to_array()

def plot_landsat5_rgb_by_index(folder_path, index):
    """
    Plot an RGB image from a Landsat 5 surface reflectance .tif file using rioxarray.

    Args:
        tif_file_path (str): Path to the .tif file.
    
    Returns:
        None. Displays the RGB image.
    """
    def normalize(array):
        """
        Normalize the array to [0, 1] for better visualization after handling NaN values.
        
        Args:
            array (numpy.ndarray): Input array to normalize.
        
        Returns:
            numpy.ndarray: Normalized array.
        """
        # Replace NaN values with the minimum of the non-NaN values
        nan_mask = np.isnan(array)
        if nan_mask.any():
            array[nan_mask] = np.nanmin(array)

        # Debug: Print the min and max values of the array
        # print("Min value:", array.min(), "Max value:", array.max())

        # Normalize the array
        return (array - array.min()) / (array.max() - array.min())
    
    # Open the .tif file with rioxarray
    image_xr = read_tif_by_index(folder_path, index)
    image_xr = image_xr.to_array()
    
    # Ensure the file has at least 3 bands (RGB)
    # if image_xr.rio.count < 3:
    #     raise ValueError(f"The .tif file must have at least 3 bands for RGB. Found {image_xr.rio.count} bands.")
    
    # Select the Red, Green, and Blue bands (3, 2, 1 for Landsat 5)
    red_band = image_xr.isel(band=4)  # Band index is 0-based
    green_band = image_xr.isel(band=3)
    blue_band = image_xr.isel(band=2)

    # Normalize each band
    red = normalize(red_band.values)
    green = normalize(green_band.values)
    blue = normalize(blue_band.values)

    # Stack the bands into an RGB image
    rgb = np.stack([red, green, blue], axis=-1)

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb[0])
    plt.axis('off')  # Remove axis for better visualization
    plt.title("Landsat 5 RGB Image")
    plt.show()

def count_observations_by_month(parent_folder):
    """
    Counts the number of observations (TIF files) per month for each year across subfolders
    and outputs a CSV file.
    
    Args:
        parent_folder (str): Path to the parent folder (e.g., YOLO).
    
    Returns:
        pd.DataFrame: A DataFrame with years as rows, months as columns, and counts of observations.
    """
    # Dictionary to store counts with (year, month) as key
    observations = defaultdict(int)
    
    # Traverse subfolders
    for year_folder in sorted(os.listdir(parent_folder)):
        year_path = os.path.join(parent_folder, year_folder)
        # Check if the subfolder is named as a year (e.g., 1985, 1986, ...)
        if os.path.isdir(year_path) and year_folder.isdigit() and len(year_folder) == 4:
            year = int(year_folder)
            for file in os.listdir(year_path):
                if file.endswith(".tif"):
                    parts = file.split("_")
                    if len(parts) == 3:
                        date = parts[2][:-4] # Extract the date (yyyymmdd)
                        if len(date) == 8 and date.isdigit():
                            month = int(date[4:6])
                            observations[(year, month)] += 1

    # Create a DataFrame from the counts
    years = sorted(set(year for year, _ in observations.keys()))
    months = [f"{month:02}" for month in range(1, 13)]  # Columns for months
    data = {month: [observations.get((year, int(month)), 0) for year in years] for month in months}
    df = pd.DataFrame(data, index=years)
    df.index.name = "Year"
    
    # Save to CSV
    output_csv = os.path.join(parent_folder, "observations_per_month.csv")
    df.to_csv(output_csv)
    print(f"CSV file saved at: {output_csv}")
    
    return df

def rename_tif_files(folder_path):
    """
    Rename `.tif` files in the folder by removing the `id` portion.
    
    Args:
        folder_path (str): Path to the folder containing `.tif` files.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tif"):
            # Split the file name into parts and remove the ID
            parts = file_name.split("_")
            if len(parts) == 4:  # Ensure it matches the expected format
                new_name = f"{parts[0]}_{parts[1]}_{parts[3]}"
                old_path = os.path.join(folder_path, file_name)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: {file_name} -> {new_name}")

def remove_small_tif_files(parent_folder, size_threshold_mb=30):
    """
    Recursively checks subfolders for .tif files smaller than the specified size
    and removes them while printing their subfolder and file name.
    
    Args:
        parent_folder (str): Path to the parent folder (e.g., YOLO).
        size_threshold_mb (int): Size threshold in MB for file removal.
    """
    size_threshold_bytes = size_threshold_mb * 1024 * 1024  # Convert MB to bytes
    
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".tif"):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                if file_size < size_threshold_bytes:
                    # Print subfolder and file name
                    subfolder = os.path.relpath(root, parent_folder)
                    print(f"Removing: Subfolder: {subfolder}, File: {file}")
                    
                    # Remove the file
                    os.remove(file_path)

def filter_and_save_bands(folder_path, keep_bands):
    """
    Filters and keeps only the specified bands from Landsat `.tif` files
    and saves the filtered `.tif` files in the same directory.
    
    Args:
        folder_path (str): Path to the parent folder containing subfolders for each year.
        keep_bands (list): List of band indices to keep (0-based indexing for Landsat).
    
    Notes:
        - The function assumes the bands are indexed as 0-based in the TIFF files.
        - Keeps the same filename, appending `_filtered` before the extension.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tif"):
                file_path = os.path.join(root, file)
                
                try:
                    # Open the file as a DataArray
                    dataset = xr.open_dataarray(file_path)

                    # Check if the dataset has enough bands to filter
                    if dataset.shape[0] < max(keep_bands) + 1:
                        filtered_data = dataset
                    else:
                        # Filter bands by 0-based indices
                        filtered_data = dataset.isel(band=keep_bands)
                    
                    filtered_data = filtered_data.astype("float32")


                    # Drop metadata causing issues, if any
                    filtered_data.attrs = {}
                    filtered_data = filtered_data.assign_coords({"band": range(1, len(keep_bands) + 1)})

                    # Create a new filename with "_filtered" appended
                    filtered_file_path = os.path.join(
                        root, file.replace(".tif", "_filtered.tif")
                    )
                    
                    # Save the filtered data to a new file
                    filtered_data.rio.to_raster(
                        filtered_file_path, compress="LZW"
                    )
                    print(f"Filtered and saved: {filtered_file_path}")


                except Exception as e:
                    print(f"Failed to process {file}: {e}")

def visualize_et_tiff(tiff_file):
    """
    Visualize ET data from a TIFF file.

    Args:
        tiff_file (str): Path to the TIFF file.
    """
    with rasterio.open(tiff_file) as src:
        et_data = src.read(1)  # Read the first band
        bounds = src.bounds  # Get geographical bounds
        print(f"Geographical bounds: {bounds}")
        print(f"Data shape: {et_data.shape}")

        # Target CRS
        target_crs = "EPSG:3857"
        
        # Calculate the transform and new dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        
        # Prepare a new array to store reprojected data
        reprojected_data = np.empty((height, width), dtype=et_data.dtype)
        
        # Perform the reprojection
        reproject(
            source=et_data,
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        
        # print(f"Reprojected Geographical bounds: {transform * (0, 0, width, height)}")
        print(f"Reprojected Data shape: {reprojected_data.shape}")

    plt.figure(figsize=(10, 8))
    plt.imshow(reprojected_data, cmap='YlGnBu', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    plt.colorbar(label="Evapotranspiration (mm)")
    plt.title("Evapotranspiration (ET)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()