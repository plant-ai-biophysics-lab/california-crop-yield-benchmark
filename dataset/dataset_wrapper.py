import os
import glob
import numpy
import pandas
import geopandas
import xarray
import rioxarray
import geojson
import ee
import geemap
import json
from datetime import datetime, date, timedelta
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import rasterize
from rasterio.transform import from_origin, from_bounds
from rasterio.mask import mask
from rasterio.merge import merge
import requests
from typing import List, Union
import zipfile
from collections import defaultdict
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import pydaymet as daymet
import calendar
from skimage.transform import resize
from rapidfuzz import process, fuzz  
from concurrent.futures import ThreadPoolExecutor
import re
import shutil



# Defults Directries ; 

CA_COUNTIES_SHAPEFILE_DIR = '/data2/hkaman/Data/YieldBenchmark/SHPs/California_Counties.geojson'
CA_SHAPEFILE_DIR = '/data2/hkaman/Data/YieldBenchmark/SHPs/ca-state.csv'
DEFAULT_CRS = 4326
SAVE_ROOT_DIR = '/data2/hkaman/Data/YieldBenchmark'
YIELD_RAW_FILES_DIR = "/data2/hkaman/Data/YieldBenchmark/YieldObservation"
 

CDL_CROP_LEGEND = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",     
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
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
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes", 
    55: "Caneberries", 
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",         
    61: "Fallow/Idle Cropland",
    62: "Non-agricultural",
    63: "Non-agricultural",
    64: "Non-agricultural",
    65: "Non-agricultural",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
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
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
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
    228: "Dbl Crop Triticale/Corn", 
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht", 
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley", 
    234: "Dbl Crop Durum Wht/Sorghum", 
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton", 
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans", 
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants", 
    249: "Gourds",
    250: "Cranberries", 
    254: "Dbl Crop Barley/Soybeans",
    121: "Urban",
    141: "Water",
    0: "Other"
}

CDL_COLORS = {
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

CROP_NAME_MANUAL_MATCHES = {
            "beets garden": "Sugarbeets",
            "berries blackberries": "Caneberries",
            "berries bushberries unspecified": "Caneberries",
            "berries raspberries": "Caneberries",
            "brussels sprouts": "Brussels Sprouts",
            "cherimoyas": "Other Tree Crops",
            "chestnuts": "Other Tree Crops",
            "cilantro": "Herbs",
            "dates": "Other Tree Crops",
            "endive all": "Greens",
            "escarole all": "Greens",
            "field crop by-products": "Other Crops",
            "field crops seed misc.": "Other Crops",
            "field crops unspecified": "Other Crops",
            "figs dried": "Other Tree Crops",
            "flowers decorative dried": "Other Crops",
            "fruits & nuts unspecified": "Other Tree Crops",
            "grapefruit all": "Citrus",
            "guavas": "Other Tree Crops",
            "hay grain": "Other Hay/Non Alfalfa",
            "hay green chop": "Other Hay/Non Alfalfa",
            "hay sudan": "Other Hay/Non Alfalfa",
            "hay wild": "Other Hay/Non Alfalfa",
            "horseradish": "Other Crops",
            "jojoba": "Other Tree Crops",
            "kale": "Greens",
            "kiwifruit": "Other Tree Crops",
            "kohlrabi": "Greens",
            "kumquats": "Citrus",
            "leeks": "Onions",
            "limes all": "Citrus",
            "macadamia nuts": "Other Tree Crops",
            "melons crenshaw": "Cantaloupes",
            "melons unspecified": "Other Crops",
            "mushrooms": "Other Crops",
            "nursery plants strawberry": "Strawberries",
            "okra": "Other Crops",
            "parsnips": "Other Crops",
            "pasture forage misc.": "Other Hay/Non Alfalfa",
            "persimmons": "Other Tree Crops",
            "quince": "Other Tree Crops",
            "radicchio": "Greens",
            "rappini": "Greens",
            "rutabagas": "Greens",
            "seed bermuda grass": "Sod/Grass Seed",
            "seed sudan grass": "Other Hay/Non Alfalfa",
            "seed vegetable & vinecrop": "Other Crops",
            "silage": "Other Hay/Non Alfalfa",
            "spinach food service": "Greens",
            "spinach fresh market": "Greens",
            "spinach unspecified": "Greens",
            "straw": "Other Hay/Non Alfalfa",
            "sugar beets": "Sugarbeets",
            "swiss chard": "Greens",
            "tangelos": "Citrus",
            "tangerines & mandarins": "Citrus",
            "vegetables baby": "Misc Vegs & Fruits",
            "vegetables oriental all": "Misc Vegs & Fruits",
            "vegetables unspecified": "Misc Vegs & Fruits",
            "wheat seed": "Winter Wheat",
            "berries boysenberries": "Caneberries",
            "berries loganberries": "Caneberries",
            "chayotes": "Misc Vegs & Fruits",
            "taro root": "Other Crops",
            "seed clover unspecified": "Clover/Wildflowers",
            "ryegrass perennial all": "Other Crops",
            "flowers lilacs cut": "Other Crops",
            "yucca": "Other Crops",
            "nursery fruit/vine/nut non-bearing": "Other Tree Crops",
            "beets": "Sugarbeets",
            "figs": "Other Tree Crops",
            "vegetables asian": "Misc Vegs & Fruits",
            "alfalfa, silage": "Alfalfa",
            "anise/fennel": "Herbs",
            "barley, grain": "Barley",
            "barley, misc uses": "Barley",
            "beans, all": "Dry Beans",
            "beans, fresh (snap)": "Dry Beans",
            "beans, misc, including chickpeas": "Dry Beans",
            "berries, blackberries": "Caneberries",
            "berries, raspberries": "Caneberries",
            "berries, strawberries, all": "Strawberries",
            "berries, strawberries, misc": "Strawberries",
            "bok choy": "Greens",
            "chard": "Greens",
            "cilantro": "Herbs",
            "citrus, misc": "Citrus",
            "corn, sweet (fresh)": "Sweet Corn",
            "cotton, lint, all": "Cotton",
            "cotton, lint, misc": "Cotton",
            "cotton, lint, pima": "Cotton",
            "cotton, lint, upland": "Cotton",
            "dates": "Other Tree Crops",
            "endive": "Greens",
            "escarole": "Greens",
            "figs": "Other Tree Crops",
            "grapefruit": "Citrus",
            "grapes, misc": "Grapes",
            "grapes, raisin": "Grapes",
            "grapes, wine, all": "Grapes",
            "grapes, wine, misc": "Grapes",
            "grapes, wine, red": "Grapes",
            "grapes, wine, white": "Grapes",
            "green chop": "Other Hay/Non Alfalfa",
            "greens, specialty": "Greens",
            "hay, bermuda grass": "Other Hay/Non Alfalfa",
            "hay, grain, misc": "Other Hay/Non Alfalfa",
            "hay, misc": "Other Hay/Non Alfalfa",
            "hay, sudan": "Other Hay/Non Alfalfa",
            "hay, wild": "Other Hay/Non Alfalfa",
            "horseradish": "Other Crops",
            "hybrid stone fruits": "Other Tree Crops",
            "jujubes": "Other Tree Crops",
            "kale": "Greens",
            "kiwifruit": "Other Tree Crops",
            "kumquats": "Citrus",
            "leeks": "Onions",
            "melons, misc": "Cantaloupes",
            "melons, watermelon": "Watermelons",
            "mushrooms": "Other Crops",
            "oats, grain": "Oats",
            "oats, hay": "Oats",
            "oats, silage": "Oats",
            "okra": "Other Crops",
            "oranges, navel": "Oranges",
            "oranges, valencia": "Oranges",
            "peaches, clingstone": "Peaches",
            "peaches, freestone": "Peaches",
            "pears, all": "Pears",
            "pears, asian": "Pears",
            "pears, bartlett": "Pears",
            "pears, misc": "Pears",
            "peas, green (fresh)": "Peas",
            "persimmons": "Other Tree Crops",
            "quince": "Other Tree Crops",
            "rice, all": "Rice",
            "rice, excluding wild": "Rice",
            "rice, wild": "Rice",
            "ryegrass": "Other Crops",
            "ryegrass, all": "Other Crops",
            "seed for planting, bean": "Dry Beans",
            "seed for planting, bermuda grass": "Sod/Grass Seed",
            "seed for planting, potato": "Potatoes",
            "seed for planting, wheat": "Winter Wheat",
            "silage, misc": "Other Hay/Non Alfalfa",
            "sorghum, silage": "Sorghum",
            "squash, misc": "Squash",
            "squash, summer": "Squash",
            "tangelos": "Citrus",
            "tangerines & mandarins": "Citrus",
            "tomatoes, fresh": "Tomatoes",
            "triticale, misc uses": "Triticale",
            "wheat, hay": "Winter Wheat",
            "wheat, misc uses": "Winter Wheat",
            "berries, misc": "Caneberries",
            "oats, misc uses": "Oats",
            "peppers, chili": "Peppers",
            "seed for planting, misc": "Other Crops",
            "seed for planting, misc field crops": "Other Crops",
            "anise (fennel)": "Herbs",
            "artichokes": "Misc Vegs & Fruits",
            "asparagus": "Asparagus",
            "barley": "Barley",
            "beans, dry edible": "Dry Beans",
            "beans, unspecified": "Dry Beans",
            "broccoli": "Broccoli",
            "cabbage": "Cabbage",
            "carrots": "Carrots",
            "cauliflower": "Cauliflower",
            "celery": "Celery",
            "cherries": "Cherries",
            "cotton": "Cotton",
            "corn, grain": "Corn",
            "corn, silage": "Corn",
            "corn, sweet": "Sweet Corn",
            "cucumbers": "Cucumbers",
            "eggplant": "Eggplants",
            "garlic": "Garlic",
            "grapes, table": "Grapes",
            "grapes, wine": "Grapes",
            "greens, leafy": "Greens",
            "hops": "Hops",
            "lettuce, head": "Lettuce",
            "lettuce, leaf": "Lettuce",
            "lettuce, romaine": "Lettuce",
            "melons, cantaloup": "Cantaloupes",
            "melons, honeydew": "Honeydew Melons",
            "nectarines": "Nectarines",
            "oats": "Oats",
            "olives": "Olives",
            "onions": "Onions",
            "peaches": "Peaches",
            "pears": "Pears",
            "parsley": "Herbs",
            "peas, green": "Peas",
            "peppers, bell": "Peppers",
            "pistachios": "Pistachios",
            "plums": "Plums",
            "pomegranates": "Pomegranates",
            "potatoes": "Potatoes",
            "pumpkins": "Pumpkins",
            "radishes": "Radishes",
            "rice": "Rice",
            "rye": "Rye",
            "spinach processing": "Greens",
            "safflower": "Safflower",
            "sorghum": "Sorghum",
            "spinach": "Greens",
            "squash": "Squash",
            "strawberries": "Strawberries",
            "sunflower": "Sunflower",
            "sweet potatoes": "Sweet Potatoes",
            "tomatoes, processing": "Tomatoes",
            "tomatoes, fresh market": "Tomatoes",
            "turnips": "Turnips",
            "walnuts": "Walnuts",
            "watermelons": "Watermelons",
            "wheat, durum": "Durum Wheat",
            "wheat, spring": "Spring Wheat",
            "wheat, winter": "Winter Wheat"
        }

#***********************************************#
#************* Download Modules ****************#
#***********************************************#

class DownloadCDLEE():
    def __init__(self, 
                 year: int, 
                 county_name: str):
        
        self.year = year
        self.county_name = county_name

        # Load California counties
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=DEFAULT_CRS)

    def __call__(self):

        self.get_cdl_data_for_county()

    def get_cdl_data_for_county(self):

        geometry, _, ee_geometry, county_name_modified = get_county_info(self.dataframe, self.county_name)
        cdl_by_year = self.get_cdl_data()
        cdl_county = self.crop_cdl_to_geometry(cdl_by_year, geometry)
        self.export_image(cdl_county, county_name_modified, ee_geometry)

        # return cdl_county

    def get_cdl_data(self):
        # Load CDL data for the specified year
        start_date, end_date = get_start_end_year_dates(self.year)
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
        # self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = geopandas.read_file(CA_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=DEFAULT_CRS)

    def __call__(self):
        return self.get_climate_data_county()
    

    def get_climate_data_county(self):
        # _, geometry, _, county_name_modified = get_county_info(self.dataframe, self.county_name)
        polygon = self.dataframe.iloc[0].geometry
        aoi_geojson = geojson.Feature(geometry=mapping(polygon))
        geometry = aoi_geojson["geometry"]


        dataset = self.get_daymet_data(geometry)
        county_name_modified = 'CA'
        self.export_image(dataset, county_name_modified)

        return dataset

    
    def get_daymet_data(self, geometry):

        start_date, end_date = get_start_end_year_dates(self.year)
        dates = (start_date, end_date)
  
        # variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'snow', 'vp', 'pet'] #["prcp", "tmin", "tmax", "vp", "srad"]
        dataset = daymet.get_bygeom(geometry, dates, pet="priestley_taylor", snow=True)

        return dataset
    def export_image(self, dataset, county_name):
        year_folder = str(self.year)

        dir_path = os.path.join(SAVE_ROOT_DIR, f"{county_name}/Climate", year_folder)
        os.makedirs(dir_path, exist_ok=True)

        folder_name = f"DayMet_{county_name}_{year_folder}.nc"
        full_name = os.path.join(dir_path, folder_name)

        # Define compression settings
        comp = dict(zlib=True, complevel=5)  # Maximum compression level (0-9)
        encoding = {var: comp for var in dataset.data_vars}
        dataset.to_netcdf(full_name, encoding=encoding)

        print(f"NetCDF file saved to {full_name}")

class DownloadOpenETEE:
    def __init__(self, county_name: str, year: int):

        self.year = year
        self.county_name = county_name

        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=DEFAULT_CRS)

    def __call__(self):
        """
        Downloads OpenET data for each month in the specified year for the given geometry.
        """
        geometry, _, ee_geometry, county_name_modified = get_county_info(self.dataframe, self.county_name)

        for i in range(12):
            start_date, end_date = get_monthly_dates(self.year, i)
            print(start_date, end_date)
            image = self.get_openet_image(geometry, ee_geometry, start_date, end_date)
            if image is not None:
                self.export_image(image, county_name_modified, ee_geometry, index=i)


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

    def export_image_to_local(self, image, county_name, ee_geometry, index=None):

        image = image.toFloat()
        file_path = '/data2/hkaman/Data/FoundationModel/ETS/'

        try:
            image_id = image.get('system:id').getInfo()
            file_name_part = image_id.split('/')[-1]
        except Exception:
            file_name_part = f"image_{index}"

        file_name = f"{county_name}_{file_name_part}.tif"
        full_path = f"{file_path}{file_name}"

        print(f"Downloading {file_name} to {full_path}...")

        ## Estimate Image Size in MB
        # num_pixels = image.clip(ee_geometry).reduceRegion(
        #     reducer=ee.Reducer.count(),
        #     geometry=ee_geometry,
        #     scale=30,  # Use the same scale as export
        #     maxPixels=1e13
        # ).values().get(0)

        # num_bands = image.bandNames().size()
        # bit_depth = 16  # Landsat images are usually 16-bit

        # # Compute file size in MB
        # size_mb = ee.Number(num_pixels).multiply(num_bands).multiply(bit_depth).divide(8 * 1024**2).getInfo()

        # print(f"Estimated image size: {size_mb:.2f} MB")


        geemap.ee_export_image(
            image.clip(ee_geometry), 
            filename=full_path, 
            scale=30, 
            region=ee_geometry, 
            file_per_band=False,
        )

        print(f"Image saved to {full_path}")

class SSURGODownloader:
    def __init__(self, county_name: str, year: int):
        self.county_name = county_name
        self.year = year 

        # Load California counties
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=DEFAULT_CRS)

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
        geometry, _, county_name_modified = get_county_info(self.dataframe, self.county_name)


        # if isinstance(geometry, geopandas.GeoDataFrame):
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
        if isinstance(geometry, geopandas.GeoDataFrame):
            geometry = geometry.geometry.unary_union
        geometry_dict = mapping(geometry)
        geometry_json = json.dumps(geometry_dict)
        return geometry_json

class DownloadSatelliteImgEE:
    def __init__(self, 
                 year: int, 
                 county_name:str,
                 start_date: str, 
                 end_date: str,
                 cloud_filter: float, 
                 satellite: str):
        self.year = year
        self.county_name = county_name
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_filter = cloud_filter
        self.satellite = satellite

        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg = DEFAULT_CRS)

    def __call__(self):
        self.get_satellite_data_for_county()

    def get_satellite_data_for_county(self):

        geometry, _, ee_geometry, county_name_modified = get_county_info(self.dataframe, self.county_name)

        images = self.get_landsat_images_by_month(geometry, ee_geometry, self.start_date, self.end_date)

        # for image in images:
        if images is not None:
            self.export_image(images, county_name_modified, ee_geometry, index=0)


    
    def get_image_size(self, image):

        num_pixels = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=image.geometry(),
            scale=30,  
            maxPixels=1e13
        ).values().get(0) 

        num_bands = image.bandNames().size()
        bit_depth = 16
        size_mb = ee.Number(num_pixels).multiply(num_bands).multiply(bit_depth).divide(8 * 1024**2)
        return size_mb

    def has_missing_values(self, image):

        mask = image.mask()
        missing = mask.Not() 
        missing_count = missing.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=image.geometry(),
            scale=30,
            maxPixels=1e13
        ).getInfo()

        return any(value > 0 for value in missing_count.values())
    
    def fill_large_gaps_with_mosaic(self, collection, geometry):

        combined_image = collection.mosaic().clip(geometry)
        return combined_image
    
    def fill_with_focal_mean(self, image):

        return image.unmask(None).focal_mean(radius=9, units='pixels', kernelType='square')

    def get_landsat_images_by_month(self, geometry, ee_geometry, start_date, end_date):

        landsat5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
        landsat7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'])
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])


        landsat = (landsat5.merge(landsat7).merge(landsat8)
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                )
        print(f"Before cloud masking {landsat.size().getInfo()} were found!")
        image_list = landsat.toList(landsat.size())
        sorted_landsat = landsat.sort('CLOUD_COVER')
        best_image = None
        image_list = sorted_landsat.toList(sorted_landsat.size())

        list_images = []

        for i in range(sorted_landsat.size().getInfo()):
            image = ee.Image(image_list.get(i))
            size_mb = self.get_image_size(image).getInfo()
            image = self.crop_image_to_geometry(image, ee_geometry)
            list_images.append(image)
            if size_mb >= 100:
                best_image = image
                # best_image = self.fill_missing_values(best_image)
                print(f"Selected image with cloud cover: {best_image.get('CLOUD_COVER').getInfo()} and size: {size_mb:.2f} MB")
                break
            else:
                print(f"Image skipped due to size: {size_mb:.2f} MB")

        if best_image is None:
            print("No suitable Landsat image found for the specified month.")

        print(f"Selected image ID: {best_image.get('system:id').getInfo()} with cloud cover: {best_image.get('CLOUD_COVER').getInfo()}")
        return best_image

    def crop_image_to_geometry(self, image, geometry):

        return image.clip(geometry)

    def export_image(self, image, county_name, ee_geometry, index=None):
        image = image.toFloat()

        try:
            image_id = image.get('system:id').getInfo()
            file_name_part = image_id.split('/')[-1]
        except Exception:
            file_name_part = f"image_{index}"  

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
    
    def export_image_to_local(self, image, county_name, ee_geometry, index=None):

        image = image.toFloat()
        file_path = '/data2/hkaman/Data/FoundationModel/Landsat/'

        try:
            image_id = image.get('system:id').getInfo()
            file_name_part = image_id.split('/')[-1]
        except Exception:
            file_name_part = f"image_{index}"

        file_name = f"{county_name}_{file_name_part}.tif"
        full_path = f"{file_path}{file_name}"

        print(f"Downloading {file_name} to {full_path}...")

        geemap.ee_export_image(
            image, 
            filename=full_path, 
            scale=30, 
            region=ee_geometry, 
            file_per_band=False
        )

        print(f"Image saved to {full_path}")


#***********************************************#
#************* Yield Observation ***************#
#***********************************************#


class ProcessingYieldObs():
    def __init__(self, county_name: str, output_root_dir:str):

        self.county_name = county_name
        self.output_root_dir = output_root_dir

        self.crop_names = list(CDL_CROP_LEGEND.values())

        
    def __call__(self):

        # root_folder = "/data2/hkaman/Data/FoundationModel"  
        all_dataframes = [] 
        
        for file_name in sorted(os.listdir(YIELD_RAW_FILES_DIR)):
            if file_name.endswith(".csv"):
                
                year = file_name.split("_")[-1].split(".")[0]

                
                file_path = os.path.join(YIELD_RAW_FILES_DIR, file_name)
                df = pandas.read_csv(file_path)
    
                df = self.rename_column(df)

                df = self.match_crop_names(df)

                
                df.columns = df.columns.str.strip()
                
                df["key_crop_name"] = df.apply(lambda row: self.fix_no_match(row["crop_name"], row["key_crop_name"]), axis=1)
                df.loc[df["crop_name"].str.strip().str.lower() == "parsley", "key_crop_name"] = "Herbs"
                
                if self.county_name:
                    filtered_df = df[df['county'].str.strip().eq(self.county_name)]
                    
                    output_folder = os.path.join(self.output_root_dir, self.county_name, 'InD', year)
                    os.makedirs(output_folder, exist_ok=True)  

                    output_file = os.path.join(output_folder, f"yield_{year}.csv")

                    filtered_df.to_csv(output_file, index=False)

                    print(f"Processed and saved: {output_file}")

                else:
                    
                    all_dataframes.append(df)

        
        if not self.county_name and all_dataframes:
            return pandas.concat(all_dataframes, ignore_index=True)

    def fix_no_match(self, crop_name, key_crop_name):
        """
        If key_crop_name is "No Match", replace it with the corresponding manual match.
        Ensures formatting consistency between crop_name and manual_matches.
        """
        if key_crop_name != "No Match":
            return key_crop_name  

        crop_name_clean = self.clean_text(crop_name).strip()  

       
        manual_matches_cleaned = {self.clean_text(k): v for k, v in CROP_NAME_MANUAL_MATCHES.items()}

        
        return manual_matches_cleaned.get(crop_name_clean, "No Match")
    
    def rename_column(self, df):
        
        df.columns = df.columns.str.strip()

        
        if 'Commodity Code' in df.columns and 'Crop Name' in df.columns: 
            
            columns_to_drop = ['Commodity Code', 'County Code']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            
            df = df.rename(columns={
                'Year': 'year',
                'Crop Name': 'crop_name',
                'County': 'county',
                'Harvested Acres': 'harvested_acres',
                'Production': 'production',
                'Price (Dollars/Unit)': 'price_per_unit',
                'Value (Dollars)': 'value',
                'Yield': 'yield',
            })

        elif 'Current Item Name' in df.columns and 'County Code' in df.columns:  # 2021 and 2022 format
            # Drop unnecessary columns
            columns_to_drop = ['Current Item Code', 'Legacy Item Name', 'Legacy Commodity Code',
                            'County Code', 'Row Type Id', 'Commodities In Group', 'Footnote']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Rename columns
            df = df.rename(columns={
                'Year': 'year',
                'Current Item Name': 'crop_name',
                'County': 'county',
                'Harvested Acres': 'harvested_acres',
                'Production': 'production',
                'Price Per Unit': 'price_per_unit',
                'Value': 'value',
                'Yield': 'yield',
            })

        else:
            raise ValueError("Unrecognized column format. Please check the dataset.")

        # Define the new column order
        new_column_names = [
            'year', 'crop_name', 'county',
            'harvested_acres', 'production', 'price_per_unit',
            'value', 'yield'
        ]

        # Reorder the columns (only include those present in the DataFrame)
        df = df[[col for col in new_column_names if col in df.columns]]

        return df
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase and remove extra spaces
        return text.lower().strip()

    def find_best_match(self, crop_name, choices, threshold=70):
        if not isinstance(crop_name, str):
            return "No Match"
        crop_name_clean = self.clean_text(crop_name).strip()
        choices_clean = [self.clean_text(choice) for choice in choices]

        result = process.extractOne(crop_name_clean, choices_clean, scorer=fuzz.token_set_ratio)
        if result is None:  
            return "No Match"
        match, score = choices[result[2]], result[1]  
        return match if score >= threshold else "No Match"

    def match_crop_names(self, df):
        df = df[df['county'] != 'State Total']
        df['key_crop_name'] = df['crop_name'].apply(lambda x: self.find_best_match(x, self.crop_names))
        df['yield'] = df['yield'].replace(r'^\s*$', None, regex=True)
        df = df.dropna(subset=['yield'])
        
        return df

#***********************************************#
#***************** Data Creator ****************#
#***********************************************#

class ModelProcessedDataModified:

    def __init__(self, county_name: str = 'Monterey', year: Union[List[int], int] = 2008, crop_names: Union[str, List[str], None] = None):
        """
        Args:
            county_name (str): The name of the county (e.g., 'Monterey').
            year (list[int] or int): The year or list of years for data (e.g., 2008 or [2008, 2009]).
            crop_names (str, list[str], or None): Name(s) of the crop(s) to filter (e.g., 'Corn' or ['Corn', 'Soybeans']).
        """
        self.county_name = county_name
        self.year = year if isinstance(year, list) else [year]  # Ensure year is a list
        self.crop_names = crop_names
        self.target_crs = "EPSG:32610"  # UTM Zone 10N
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=int(self.target_crs.split(':')[-1]))

        # Base directory
        root_dir = '/data2/hkaman/Data/YieldBenchmark/counties'
        self.output_dir = os.path.join(root_dir,county_name, f"InD/{self.year[0]}/{county_name}_{self.year[0]}")

        # Paths for each year
        self.cdl_paths = [os.path.join(root_dir, county_name, f"Raw/CDL/{yr}/{county_name}_CDL_{yr}.tif") for yr in self.year]
        self.landsat_dirs = [os.path.join(root_dir, county_name, f"Raw/Landsat/{yr}/") for yr in self.year]
        self.landsat_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.landsat_dirs)
        }
        self.et_dirs = [os.path.join(root_dir, county_name, f"Raw/ET/{yr}/") for yr in self.year]
        self.et_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.et_dirs)
        }
        self.climate_paths = [os.path.join(root_dir, county_name, f"Raw/Climate/{yr}/{county_name}_DayMet_{yr}.nc") for yr in self.year]
        import glob
        self.soil_dir = glob.glob(os.path.join(root_dir, county_name, "Raw/Soil/spatial/soilmu_a_*.shp"))

        if self.soil_dir:
            self.soil_dir = self.soil_dir[0]  # Take the first matching file
        else:
            print("No shapefile found.")
    
    def __call__(self, output_type: str | List[str] = "all", daily_climate: bool = True):

        if output_type == "all":
            requested_outputs = ["landsat_data", "et_data", "climate_data", "soil_data"]
        elif isinstance(output_type, str):
            requested_outputs = [output_type]
        elif isinstance(output_type, list):
            requested_outputs = output_type
        else:
            raise ValueError("Invalid output_type. Must be 'all', a string, or a list of strings.")


        valid_outputs = {"landsat_data", "et_data", "climate_data", "soil_data"}
        invalid_outputs = [key for key in requested_outputs if key not in valid_outputs]
        if invalid_outputs:
            raise ValueError(f"Invalid output(s) requested: {invalid_outputs}. Valid options are {valid_outputs}.")


        outputs = {}

        landsat_images = self.read_landsat_images()
        et_images = self.read_et_images()

        for crop_name in self.crop_names:
            cdl_cultivated_data = self.get_cultivated_area(crop_name = crop_name)
            print(f"{crop_name} | {numpy.sum(cdl_cultivated_data)}")
            if  numpy.sum(cdl_cultivated_data)> 10000: 
                crop_outputs = {}
                landsat_vector = self.get_masked_landsat_timeseries(landsat_images, cdl_cultivated_data)
                et_vector = self.get_masked_et_timeseries(et_images, cdl_cultivated_data)
                climate_vector = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
                soil_vector = self.get_soil_dataset(cdl_cultivated_data)

                if any(x is None for x in [landsat_vector, et_vector, climate_vector, soil_vector]):
                    print(f"Skipping {crop_name} due to missing data")
                    continue  # Skip this crop


                crop_outputs["landsat_data"] = landsat_vector
                crop_outputs["et_data"] = et_vector
                crop_outputs["climate_data"] = climate_vector
                crop_outputs["soil_data"] = soil_vector

                print(landsat_vector.shape, et_vector.shape, climate_vector.shape, soil_vector.shape)
                    # if "landsat_data" in requested_outputs:
                    #     landsat_vector = self.get_masked_landsat_timeseries(landsat_images, cdl_cultivated_data)
                    #     crop_outputs["landsat_data"] = landsat_vector
                    #     print(landsat_vector.shape)


                    # if "et_data" in requested_outputs:
                    #     et_vector = self.get_masked_et_timeseries(et_images, cdl_cultivated_data)
                    #     crop_outputs["et_data"] = et_vector
                    #     print(et_vector.shape)


                    # if "climate_data" in requested_outputs:
                    #     climate_vector = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
                    #     crop_outputs["climate_data"]  = climate_vector
                    #     print(climate_vector.shape)

                    # if "soil_data" in requested_outputs:
                    #     soil_vector = self.get_soil_dataset(cdl_cultivated_data)
                    #     crop_outputs["soil_data"]  = soil_vector
                    #     print(soil_vector.shape)


                outputs[crop_name] = crop_outputs

            
        numpy.savez_compressed(self.output_dir , inumpyut = outputs)

        return outputs

    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        if self.county_name == 'ContraCosta':
            new_name = 'Contra Costa'
        elif self.county_name == 'SanBenito':
            new_name = 'San Benito'
        elif self.county_name == 'SanDiego':
            new_name = 'San Diego'
        elif self.county_name == 'SanJoaquin':
            new_name = 'San Joaquin'
        else:
            new_name = self.county_name

        bounds = self.dataframe[self.dataframe["NAME"] == new_name + " County"].total_bounds
        resolution = (30, 30)  # 30m resolution as Landsat uses
        return bounds, resolution
    
    def read_landsat_images(self):
        bounds, resolution = self.get_reference_grid()
        landsat_files = self.landsat_files[self.year[0]]

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda path: self.process_landsat_image(path, bounds, resolution), landsat_files))
        
        # Stack aligned images into a single numpy array
        return numpy.stack(results, axis=0)

    def process_landsat_image(self, landsat_path, bounds, resolution):
        """ Read and align a single Landsat image """
        with rasterio.open(landsat_path) as src:
            landsat_data = src.read().astype(numpy.float32)  # Convert to float32
            landsat_transform = src.transform
            landsat_crs = src.crs

            # Check if reprojection is needed
            if landsat_crs == self.target_crs and landsat_transform.almost_equals(self.reference_transform, precision=1e-6):
                return landsat_data  # Skip reprojection if already aligned

            aligned_landsat, _, _ = self.align_to_geometry(
                reference_bounds=bounds,
                reference_crs=self.target_crs,
                resolution=resolution,
                source_data=landsat_data,
                source_transform=landsat_transform,
                source_crs=landsat_crs
            )

        return aligned_landsat.astype(numpy.float32)  # Ensure final output is also float32

    def align_to_geometry(self, reference_bounds, reference_crs, resolution, source_data, source_transform, source_crs):
        """ Align source data to match reference grid CRS, resolution, and bounds. """
        min_x, min_y, max_x, max_y = reference_bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        reference_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Preallocate array for efficiency
        aligned_data = numpy.empty((source_data.shape[0], height, width), dtype=source_data.dtype)

        for band_index in range(source_data.shape[0]):
            reproject(
                source=source_data[band_index],
                destination=aligned_data[band_index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear
            )

        return aligned_data, reference_transform, (height, width)

    def get_cultivated_area(self, crop_name):
        """
        Extract cultivated area for a specific crop and align it to the reference grid created from the county geometry.
        """
        with rasterio.open(self.cdl_paths[0]) as cdl_src:
            cdl_data = cdl_src.read(1)
            cdl_transform = cdl_src.transform
            cdl_crs = cdl_src.crs

            # Ensure crop_name is a valid string
            if not isinstance(crop_name, str):
                raise ValueError("Crop name must be a string.")

            
            # Get the corresponding crop code
            crop_code = next((code for code, name in CDL_CROP_LEGEND.items() if name == crop_name), None)

            # if crop_code is None:
            #     raise ValueError(f"Invalid crop name: {crop_name}. Please check the crop legend.")
            if crop_code is not None:
                # Create a mask for the selected crop

                mask = cdl_data == crop_code
                cultivated_area = numpy.where(mask, cdl_data, 0)

                # Align cultivated area to reference grid
                target_bounds, target_resolution = self.get_reference_grid()
                aligned_cultivated_area, _, _ = self.align_to_geometry(
                    reference_bounds = target_bounds,
                    reference_crs = self.target_crs,
                    resolution = target_resolution,
                    source_data = numpy.expand_dims(cultivated_area, axis=0),
                    source_transform = cdl_transform,
                    source_crs = cdl_crs
                )

                return aligned_cultivated_area
    
    def get_masked_landsat_timeseries(self, landsat, cultivated_area):
        """
        Mask Landsat timeseries imagery using the cultivated area and return non-zero pixels as a 3D matrix (T, B, N).
        """
        vector_timeseries = []
   
        for idx in range(12):
            aligned_landsat = landsat[idx]
            mask = cultivated_area > 0  
            masked_vector_landsat = aligned_landsat[:, mask[0, ...]]  
            masked_vector_landsat = numpy.nan_to_num(masked_vector_landsat, nan=0) 

            vector_timeseries.append(masked_vector_landsat)  
 
        vector_timeseries_output = stratified_sampling(numpy.stack(vector_timeseries, axis=0).astype(numpy.float32) , num_samples=128) 

        return vector_timeseries_output
    
    def process_et_image(self, et_path, bounds, resolution):

        with rasterio.open(et_path) as src:
            et_data = src.read().astype(numpy.float32)
            et_transform = src.transform
            et_crs = src.crs
            if et_crs == self.target_crs and et_transform.almost_equals(self.reference_transform, precision=1e-6):
                return et_data  
            
            aligned_et_data, _, _ = self.align_to_geometry(
                reference_bounds = bounds,
                reference_crs = self.target_crs,
                resolution = resolution,
                source_data = et_data,
                source_transform = et_transform,
                source_crs = et_crs
            )

        return aligned_et_data[0, ...].astype(numpy.float32)
    
    def read_et_images(self):
        bounds, resolution = self.get_reference_grid()
        et_files = self.et_files[self.year[0]]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda path: self.process_et_image(path, bounds, resolution), et_files))
        
        return numpy.stack(results, axis=0)
    
    def get_masked_et_timeseries(self, et_images, cultivated_area):
        """
        Mask ET timeseries imagery using the cultivated area and stack into a 4D matrix.
        """
        vector_timeseries = []

        for idx in range(12):
            aligned_et_data = et_images[idx]
            mask = cultivated_area > 0  
            vector_masked_et = aligned_et_data[mask[0, ...]]  
            vector_masked_et = numpy.nan_to_num(vector_masked_et, nan=0) 
            vector_timeseries.append(vector_masked_et)
           
        vector_output = stratified_sampling(numpy.stack(vector_timeseries, axis=0), num_samples=128)

        return vector_output#, matrix_output[:, 0, ...]

    def get_soil_dataset(self, cultivated_area):
        """
        Rasterize SSURGO shapefile, mask it with cultivated area, and extract soil attributes efficiently.
        Returns a raster dataset of shape (B, N), where B is the number of soil attributes,
        and N is the number of valid pixels (masked by cultivated_area).
        """
        # Load SSURGO polygons and project
        gdf = geopandas.read_file(self.soil_dir).to_crs(self.target_crs)

        # Get reference grid
        bounds, resolution = self.get_reference_grid()
        min_x, min_y, max_x, max_y = bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Rasterize using MUKEY
        shapes = [(geom, int(mukey)) for geom, mukey in zip(gdf.geometry, gdf["MUKEY"])]
        raster = rasterize(shapes, out_shape=(height, width), transform=transform, dtype=numpy.int32)

        # Load tabular data
        tabular_dir = self.soil_dir.rsplit("/", 1)[0].replace("spatial", "tabular")
        muaggatt_df = pandas.read_csv(f"{tabular_dir}/muaggatt.csv")

        # Normalize column names
        muaggatt_df.columns = muaggatt_df.columns.str.lower().str.strip()

        # Soil attributes of interest

        soil_attributes = [
            "aws0100wta", "slopegraddcp", "awmmfpwwta", "drclassdcd", "hydgrpdcd"
        ]
        # Ensure "mukey" exists
        if "mukey" not in muaggatt_df.columns:
            raise ValueError("MUKEY column is missing in the CSV file.")

        # Convert categorical columns to numeric
        drainage_mapping = {
            "Excessively drained": 5.0, "Well drained": 4.0, "Moderately well drained": 3.0,
            "Somewhat poorly drained": 2.0, "Poorly drained": 1.0, "Very poorly drained": 0.0
        }
        soil_groups_mapping = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}

        if "drclassdcd" in muaggatt_df.columns:
            muaggatt_df["drclassdcd"] = muaggatt_df["drclassdcd"].map(drainage_mapping).astype(numpy.float32)
        if "hydgrpdcd" in muaggatt_df.columns:
            muaggatt_df["hydgrpdcd"] = muaggatt_df["hydgrpdcd"].map(soil_groups_mapping).astype(numpy.float32)

        # Convert mukey to int for efficient indexing
        muaggatt_df["mukey"] = muaggatt_df["mukey"].astype(int)
        existing_attributes = [attr for attr in soil_attributes if attr in muaggatt_df.columns]
        # Create lookup table for soil attributes
        lookup = muaggatt_df.set_index("mukey")[existing_attributes].to_dict(orient="index")

        # Convert raster to soil attribute arrays
        mask = cultivated_area[0, ...] > 0
        N = numpy.count_nonzero(mask)

        # Convert raster to index-based lookup
        unique_mukeys = numpy.unique(raster)
        attribute_matrix = numpy.zeros((len(soil_attributes), height, width), dtype=numpy.float32)


        for i, attr in enumerate(existing_attributes):
            attr_values = numpy.array([lookup.get(mukey, {}).get(attr, 0.0) for mukey in unique_mukeys])
            attr_map = numpy.zeros_like(raster, dtype=numpy.float32)
            attr_map[raster > 0] = numpy.take(attr_values, numpy.searchsorted(unique_mukeys, raster[raster > 0]))
            attribute_matrix[i, :, :] = attr_map

        soil_maps = attribute_matrix[:, mask]
        soil_maps[numpy.isnan(soil_maps)] = 0.0

        # Stratified sampling
        vector_output = stratified_sampling(soil_maps.astype(numpy.float32), num_samples=128)

        return vector_output.astype(numpy.float32)


    def get_climate_stack(self, cultivated_area, daily: bool = True):
        """
        Processes climate data from NetCDF files, resamples to 1km resolution, and masks with cultivated area.

        Args:
            cultivated_area (numpy.ndarray): Mask for the cultivated area.
            daily (bool): If True, return daily observations (364 times). If False, return monthly means (12 times).

        Returns:
            numpy.ndarray: Climate data stack in format (T, C, H, W).
        """

        variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'vp', 'snow', 'pet']
        vector_timeseries, matrix_timeseries = [], []

        for climate_path in self.climate_paths:
            climate_data = xarray.open_dataset(climate_path)[variables]
            daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            climate_data = climate_data.rio.write_crs(daymet_crs)

            climate_data = climate_data.rio.reproject(self.target_crs , resolution=(1000, 1000))
            climate_bounds = climate_data.rio.bounds()
            # aligned_bounds = (591345.0, 3962475.0, 751815.0, 4086765.0)  # From previous calculation
            climate_data = climate_data.rio.clip_box(*climate_bounds)

            target_shape = (climate_data.rio.height, climate_data.rio.width)

            resampled_cultivated_area = self.resample_cultivated_area(cultivated_area[0, ...], target_shape)

            if not daily:
                climate_data = climate_data.groupby("time.month").mean("time")
                time_steps = climate_data.month
            else:
                time_steps = climate_data.time

            for time_step in time_steps:
                time_slice = climate_data.sel(time=time_step) if daily else climate_data.sel(month=time_step)

                climate_stack = numpy.stack(
                    [time_slice[var].values for var in variables],
                    axis=0
                ).astype(numpy.float32)

                mask = resampled_cultivated_area > 0  # Shape: (H, W)
                vector_masked_climate_stack = climate_stack[:, mask]  
                vector_masked_climate_stack = numpy.nan_to_num(vector_masked_climate_stack, nan=0) 
                vector_timeseries.append(vector_masked_climate_stack)
                vector_out = numpy.stack(vector_timeseries, axis=0)


        if numpy.stack(vector_out, axis=0).astype(numpy.float32).shape[-1] != 0:
            vector_output = stratified_sampling(numpy.stack(vector_out, axis=0).astype(numpy.float32) , num_samples=128)
            return vector_output.astype(numpy.float32)
        else: 
            return None

    def resample_cultivated_area(self, cultivated_area, target_shape):
        """
        Resample the cultivated_area to match the target shape.
        Ensures that any pixel in the resampled image is 1 if any of the higher-resolution
        contributing pixels was 1.

        Args:
            cultivated_area (numpy.ndarray): Binary mask for cultivated area (H, W).
            target_shape (tuple): Desired shape (H, W).

        Returns:
            numpy.ndarray: Resampled binary mask.
        """

        # Ensure the inumpyut is binary
        if not numpy.array_equal(cultivated_area, cultivated_area.astype(bool)):
            # raise ValueError("Inumpyut cultivated_area must already be binary.")
            cultivated_area = (cultivated_area > 0).astype(numpy.uint8)

        # Resample using maximum aggregation for binary values
        resampled_area = resize(
            cultivated_area.astype(numpy.float32),  # Use float32 for interpolation
            output_shape=target_shape,
            order=1,  # Bilinear interpolation to preserve the contribution of higher-resolution pixels
            anti_aliasing=False,
            preserve_range=True,
        )

        # Threshold: Any value > 0 means at least one contributing pixel was 1
        return (resampled_area > 0).astype(numpy.uint8)


class ModelProcessedData:

    def __init__(self, county_name: str = 'Monterey', year: Union[List[int], int] = 2008, crop_names: Union[str, List[str], None] = None):
        """
        Args:
            county_name (str): The name of the county (e.g., 'Monterey').
            year (list[int] or int): The year or list of years for data (e.g., 2008 or [2008, 2009]).
            crop_names (str, list[str], or None): Name(s) of the crop(s) to filter (e.g., 'Corn' or ['Corn', 'Soybeans']).
        """
        self.county_name = county_name
        self.year = year if isinstance(year, list) else [year]  # Ensure year is a list
        self.crop_names = crop_names
        self.target_crs = "EPSG:32610"  # UTM Zone 10N
        
        # Load California counties
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=int(self.target_crs.split(':')[-1]))

        # Base directory
        root_dir = '/data2/hkaman/Data/FoundationModel/Inputs'
        self.output_dir = os.path.join(root_dir,county_name, f"InD/{self.year[0]}/{county_name}_{self.year[0]}")

        # Paths for each year
        self.cdl_paths = [os.path.join(root_dir, county_name, f"Raw/CDL/{yr}/{county_name}_{yr}.tif") for yr in self.year]
        self.landsat_dirs = [os.path.join(root_dir, county_name, f"Raw/Landsat/{yr}/") for yr in self.year]
        self.landsat_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.landsat_dirs)
        }
        self.et_dirs = [os.path.join(root_dir, county_name, f"Raw/ET/{yr}/") for yr in self.year]
        self.et_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.et_dirs)
        }
        self.climate_paths = [os.path.join(root_dir, county_name, f"Raw/Climate/{yr}/DayMet_{county_name}_{yr}.nc") for yr in self.year]
        import glob
        self.soil_dir = glob.glob(os.path.join(root_dir, county_name, "Raw/Soil/spatial/soilmu_a_*.shp"))

        if self.soil_dir:
            self.soil_dir = self.soil_dir[0]  # Take the first matching file
        else:
            print("No shapefile found.")
    
    def __call__(self, output_type: str | List[str] = "all", daily_climate: bool = True):

        if output_type == "all":
            requested_outputs = ["landsat_data", "et_data", "climate_data", "soil_data"]
        elif isinstance(output_type, str):
            requested_outputs = [output_type]
        elif isinstance(output_type, list):
            requested_outputs = output_type
        else:
            raise ValueError("Invalid output_type. Must be 'all', a string, or a list of strings.")


        valid_outputs = {"landsat_data", "et_data", "climate_data", "soil_data"}
        invalid_outputs = [key for key in requested_outputs if key not in valid_outputs]
        if invalid_outputs:
            raise ValueError(f"Invalid output(s) requested: {invalid_outputs}. Valid options are {valid_outputs}.")


        outputs = {}
        output_matrix = {}

        for crop_name in self.crop_names:
            cdl_cultivated_data = self.get_cultivated_area(crop_name = crop_name)
            if cdl_cultivated_data is not None:

                crop_outputs = {}
                crop_outputs_matrix = {}

                if "landsat_data" in requested_outputs:
                    landsat_vector, landsat_matrix = self.get_masked_landsat_timeseries(cdl_cultivated_data)
                    crop_outputs["landsat_data"] = landsat_vector
                    # crop_outputs_matrix["landsat_data"] = landsat_matrix

                if "et_data" in requested_outputs:
                    et_vector, et_matrix = self.get_masked_et_timeseries(cdl_cultivated_data)
                    crop_outputs["et_data"] = et_vector
                    # crop_outputs_matrix["et_data"] = et_matrix

                if "climate_data" in requested_outputs:
                    climate_vector, climate_matrix = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
                    crop_outputs["climate_data"]  = climate_vector
                    # crop_outputs_matrix["climate_data"] = climate_matrix

                if "soil_data" in requested_outputs:
                    soil_vector, soil_matrix = self.get_soil_dataset(cdl_cultivated_data)
                    crop_outputs["soil_data"]  = soil_vector
                    # crop_outputs_matrix["soil_data"] = soil_matrix


                outputs[crop_name] = crop_outputs
                # output_matrix[crop_name] = crop_outputs_matrix
            
        numpy.savez_compressed(self.output_dir , inumpyut = outputs)

        return outputs#, output_matrix

    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        bounds = self.dataframe[self.dataframe["NAME"] == self.county_name + " County"].total_bounds
        resolution = (30, 30)  # 30m resolution as Landsat uses
        return bounds, resolution

    def align_to_geometry(self, reference_bounds, reference_crs, resolution, source_data, source_transform, source_crs):
        """
        Align source data to match the CRS, resolution, and bounds of the reference grid.

        Args:
            reference_bounds (tuple): Bounds of the reference grid (min_x, min_y, max_x, max_y).
            reference_crs: CRS of the reference grid.
            resolution (tuple): Resolution of the reference grid (x_res, y_res).
            source_data: Data to be aligned (numpy array).
            source_transform: Transform of the source data.
            source_crs: CRS of the source data.

        Returns:
            numpy.ndarray: Aligned data with the same resolution and extent as the reference.
        """
        min_x, min_y, max_x, max_y = reference_bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        reference_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        aligned_data = numpy.zeros((source_data.shape[0], height, width), dtype=source_data.dtype)

        for band_index in range(source_data.shape[0]):
            reproject(
                source=source_data[band_index],
                destination=aligned_data[band_index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear
            )

        return aligned_data, reference_transform, (height, width)

    def get_cultivated_area(self, crop_name):
        """
        Extract cultivated area for a specific crop and align it to the reference grid created from the county geometry.
        """
        with rasterio.open(self.cdl_paths[0]) as cdl_src:
            cdl_data = cdl_src.read(1)
            cdl_transform = cdl_src.transform
            cdl_crs = cdl_src.crs

            # Ensure crop_name is a valid string
            if not isinstance(crop_name, str):
                raise ValueError("Crop name must be a string.")

            
            # Get the corresponding crop code
            crop_code = next((code for code, name in CDL_CROP_LEGEND.items() if name == crop_name), None)

            # if crop_code is None:
            #     raise ValueError(f"Invalid crop name: {crop_name}. Please check the crop legend.")
            if crop_code is not None:
                # Create a mask for the selected crop

                mask = cdl_data == crop_code
                cultivated_area = numpy.where(mask, cdl_data, 0)

                # Align cultivated area to reference grid
                target_bounds, target_resolution = self.get_reference_grid()
                aligned_cultivated_area, _, _ = self.align_to_geometry(
                    reference_bounds = target_bounds,
                    reference_crs = self.target_crs,
                    resolution = target_resolution,
                    source_data = numpy.expand_dims(cultivated_area, axis=0),
                    source_transform = cdl_transform,
                    source_crs = cdl_crs
                )

                return aligned_cultivated_area
    
    def get_masked_landsat_timeseries(self, cultivated_area):
        """
        Mask Landsat timeseries imagery using the cultivated area and return non-zero pixels as a 3D matrix (T, B, N).
        """
        vector_timeseries, matrix_timeseries = [], []
        bounds, resolution = self.get_reference_grid()


        for landsat_path in self.landsat_files[self.year[0]]:
            with rasterio.open(landsat_path) as src:
                landsat_data = src.read()  
                landsat_transform = src.transform
                landsat_crs = src.crs

                aligned_landsat, _, _ = self.align_to_geometry(
                    reference_bounds=bounds,
                    reference_crs=self.target_crs,
                    resolution=resolution,
                    source_data=landsat_data,
                    source_transform=landsat_transform,
                    source_crs=landsat_crs
                ) 

                mask = cultivated_area > 0  
                masked_vector_landsat = aligned_landsat[:, mask[0, ...]]  
                masked_vector_landsat = numpy.nan_to_num(masked_vector_landsat, nan=0) 
                vector_timeseries.append(masked_vector_landsat)  
                 

                mask_matrix = numpy.expand_dims(mask, axis=0)
                masked_landsat = aligned_landsat * mask_matrix
                masked_landsat = numpy.nan_to_num(masked_landsat, nan=0) 
                matrix_timeseries.append(masked_landsat)


        matrix_timeseries_output = numpy.stack(matrix_timeseries, axis=0)
        vector_timeseries_output = stratified_sampling(numpy.stack(vector_timeseries, axis=0) , num_samples=128) 

        return vector_timeseries_output, matrix_timeseries_output[:, 0, ...]

    def get_masked_et_timeseries(self, cultivated_area):
        """
        Mask ET timeseries imagery using the cultivated area and stack into a 4D matrix.
        """
        matrix_timeseries, vector_timeseries = [], []
        bounds, resolution = self.get_reference_grid()

        for et_path in self.et_files[self.year[0]]:
            with rasterio.open(et_path) as src:
                et_data = src.read()
                et_transform = src.transform
                et_crs = src.crs

                aligned_et_data, _, _ = self.align_to_geometry(
                    reference_bounds = bounds,
                    reference_crs = self.target_crs,
                    resolution = resolution,
                    source_data = et_data,
                    source_transform = et_transform,
                    source_crs = et_crs
                )

                mask = cultivated_area > 0  
                vector_masked_et = aligned_et_data[:, mask[0, ...]]  
                vector_masked_et = numpy.nan_to_num(vector_masked_et, nan=0) 
                vector_timeseries.append(vector_masked_et)
                

                mask_matrix = numpy.expand_dims(mask, axis=0)
                matrix_masked_et = aligned_et_data * mask_matrix
                matrix_masked_et = numpy.nan_to_num(matrix_masked_et, nan=0) 
                matrix_timeseries.append(matrix_masked_et)


        matrix_output = numpy.stack(matrix_timeseries, axis=0)
        vector_output = stratified_sampling(numpy.stack(vector_timeseries, axis=0), num_samples=128)

        return vector_output, matrix_output[:, 0, ...]
    
    def get_soil_dataset(self, cultivated_area):
        """
        Rasterize SSURGO shapefile to match the reference grid and mask it with cultivated area.
        Extracts 10 key soil attributes from tabular data and returns a unique raster for each attribute.

        Returns:
            numpy.ndarray: A raster dataset of shape (B, N), where B is the number of soil attributes,
                        and N is the number of valid pixels (masked by cultivated_area).
        """

        # Load SSURGO spatial map unit polygons
        gdf = geopandas.read_file(self.soil_dir)
        bounds, resolution = self.get_reference_grid()
        gdf = gdf.to_crs(self.target_crs)

        # gdf["MUSYM_CODE"] = gdf["MUSYM"].astype("category").cat.codes
        # shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf["MUSYM_CODE"])]
        shapes = [(geom, int(value)) for geom, value in zip(gdf.geometry, gdf["MUKEY"])]

        min_x, min_y, max_x, max_y = bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            dtype="int32"
        )


        # Load tabular data
        tabular_dir = self.soil_dir.rsplit("/", 1)[0].replace("spatial", "tabular")
        muaggatt_df = pandas.read_csv(f"{tabular_dir}/muaggatt.csv")

        # Convert column names to lowercase and strip spaces
        muaggatt_df.columns = muaggatt_df.columns.str.lower().str.strip()

        soil_attributes = [
            "aws0100wta", "ph1to1h2o_r", "sandtotal_r", "silttotal_r", "claytotal_r", 
            "claytotal_r", "slopegraddcp", "awmmfpwwta",
            "drclassdcd", "hydgrpdcd"
        ]

        # Convert MUKEY & COKEY to string before merging
        if "mukey" in muaggatt_df.columns:
            muaggatt_df["mukey"] = muaggatt_df["mukey"].astype(str)
        if "cokey" in muaggatt_df.columns:
            muaggatt_df["cokey"] = muaggatt_df["cokey"].astype(str)

        # Check if expected soil attributes exist
        existing_attributes = [attr for attr in soil_attributes if attr in muaggatt_df.columns]
        # print("Matched Soil Attributes in Data:", existing_attributes)

        # Ensure "mukey" column exists before setting index
        if "mukey" not in muaggatt_df.columns:
            raise ValueError("MUKEY column is missing in the CSV file.")

        # Convert categorical drainage class to numerical values
        drainage_mapping = {
            "Excessively drained": 5.0,
            "Well drained": 4.0,
            "Moderately well drained": 3.0,
            "Somewhat poorly drained": 2.0,
            "Poorly drained": 1.0,
            "Very poorly drained": 0.0,
        }

        soil_groups__mapping = {
            "A": 0.0,
            "B": 1.0,
            "C": 2.0,
            "D": 3.0,
        }


        if "drclassdcd" in muaggatt_df.columns:
            muaggatt_df["drclassdcd"] = muaggatt_df["drclassdcd"].map(drainage_mapping).astype(numpy.float32)

        if "hydgrpdcd" in muaggatt_df.columns:
            muaggatt_df["hydgrpdcd"] = muaggatt_df["hydgrpdcd"].map(soil_groups__mapping).astype(numpy.float32)



        mask = cultivated_area[0, ...] > 0  # Shape: (H, W)
        N = numpy.count_nonzero(mask)  # Number of valid pixels

        soil_maps = numpy.zeros((len(existing_attributes), N), dtype=numpy.float32)
        out_matrix = []


        for i, attr in enumerate(existing_attributes):
            attr_map = numpy.zeros_like(raster, dtype=numpy.float32)  

            for _, row in muaggatt_df.iterrows():
                mu_key = row["mukey"]
                value = row.get(attr, 0.0)  
                true_indices = numpy.where(raster == numpy.float32(mu_key))
                attr_map[true_indices] = value 
            # Extract only non-zero pixels where cultivated area > 0
            soil_maps[i, :] = attr_map[mask]
            soil_maps[i, numpy.isnan(soil_maps[i, :])] = 0.0

            matrix_output = numpy.where(cultivated_area[0] > 0, attr_map, 0)
            matrix_output = numpy.nan_to_num(matrix_output, nan=0) 
            out_matrix.append(matrix_output)
            matrix_outputs = numpy.stack(out_matrix, axis=0)

        vector_output = stratified_sampling(soil_maps, num_samples=128)

        return vector_output, matrix_outputs

    def get_climate_stack(self, cultivated_area, daily: bool = True):
        """
        Processes climate data from NetCDF files, resamples to 1km resolution, and masks with cultivated area.

        Args:
            cultivated_area (numpy.ndarray): Mask for the cultivated area.
            daily (bool): If True, return daily observations (364 times). If False, return monthly means (12 times).

        Returns:
            numpy.ndarray: Climate data stack in format (T, C, H, W).
        """

        variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'vp', 'snow', 'pet']
        vector_timeseries, matrix_timeseries = [], []

        for climate_path in self.climate_paths:
            climate_data = xarray.open_dataset(climate_path)[variables]
            daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            climate_data = climate_data.rio.write_crs(daymet_crs)

            climate_data = climate_data.rio.reproject(self.target_crs , resolution=(1000, 1000))
            climate_bounds = climate_data.rio.bounds()
            # aligned_bounds = (591345.0, 3962475.0, 751815.0, 4086765.0)  # From previous calculation
            climate_data = climate_data.rio.clip_box(*climate_bounds)

            target_shape = (climate_data.rio.height, climate_data.rio.width)

            resampled_cultivated_area = self.resample_cultivated_area(cultivated_area[0, ...], target_shape)

            if not daily:
                climate_data = climate_data.groupby("time.month").mean("time")
                time_steps = climate_data.month
            else:
                time_steps = climate_data.time

            for time_step in time_steps:
                time_slice = climate_data.sel(time=time_step) if daily else climate_data.sel(month=time_step)

                climate_stack = numpy.stack(
                    [time_slice[var].values for var in variables],
                    axis=0
                )

                mask = resampled_cultivated_area > 0  # Shape: (H, W)
                vector_masked_climate_stack = climate_stack[:, mask]  
                vector_masked_climate_stack = numpy.nan_to_num(vector_masked_climate_stack, nan=0) 
                vector_timeseries.append(vector_masked_climate_stack)
                vector_out = numpy.stack(vector_timeseries, axis=0)

                mask_matrix = numpy.expand_dims(mask, axis=0)
                masked_climate_stack = climate_stack * mask_matrix
                masked_climate_stack = numpy.nan_to_num(masked_climate_stack, nan=0) 
                matrix_timeseries.append(masked_climate_stack)

        vector_output = stratified_sampling(numpy.stack(vector_out, axis=0) , num_samples=128) 
        matrix_output = numpy.stack(matrix_timeseries, axis=0)


        return vector_output, matrix_output

    def resample_cultivated_area(self, cultivated_area, target_shape):
        """
        Resample the cultivated_area to match the target shape.
        Ensures that any pixel in the resampled image is 1 if any of the higher-resolution
        contributing pixels was 1.

        Args:
            cultivated_area (numpy.ndarray): Binary mask for cultivated area (H, W).
            target_shape (tuple): Desired shape (H, W).

        Returns:
            numpy.ndarray: Resampled binary mask.
        """

        # Ensure the inumpyut is binary
        if not numpy.array_equal(cultivated_area, cultivated_area.astype(bool)):
            # raise ValueError("Inumpyut cultivated_area must already be binary.")
            cultivated_area = (cultivated_area > 0).astype(numpy.uint8)

        # Resample using maximum aggregation for binary values
        resampled_area = resize(
            cultivated_area.astype(numpy.float32),  # Use float32 for interpolation
            output_shape=target_shape,
            order=1,  # Bilinear interpolation to preserve the contribution of higher-resolution pixels
            anti_aliasing=False,
            preserve_range=True,
        )

        # Threshold: Any value > 0 means at least one contributing pixel was 1
        return (resampled_area > 0).astype(numpy.uint8)


class CountyDataCreator:

    def __init__(self, county_name: str = 'Monterey', year: Union[List[int], int] = 2008, crop_names: Union[str, List[str], None] = None):
        """
        Args:
            county_name (str): The name of the county (e.g., 'Monterey').
            year (list[int] or int): The year or list of years for data (e.g., 2008 or [2008, 2009]).
            crop_names (str, list[str], or None): Name(s) of the crop(s) to filter (e.g., 'Corn' or ['Corn', 'Soybeans']).
        """
        self.county_name = county_name
        self.year = year if isinstance(year, list) else [year]  # Ensure year is a list
        self.crop_names = crop_names
        self.target_crs = "EPSG:32610"  # UTM Zone 10N
        
        # Load California counties
        self.dataframe = geopandas.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=int(self.target_crs.split(':')[-1]))

        # Base directory
        root_dir = '/data2/hkaman/Data/FoundationModel/Inputs'

        # Paths for each year
        self.cdl_paths = [os.path.join(root_dir, county_name, f"Raw/CDL/{yr}/{county_name}_{yr}.tif") for yr in self.year]
        self.landsat_dirs = [os.path.join(root_dir, county_name, f"Raw/Landsat/{yr}/") for yr in self.year]
        self.landsat_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.landsat_dirs)
        }
        self.et_dirs = [os.path.join(root_dir, county_name, f"Raw/ET/{yr}/") for yr in self.year]
        self.et_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.et_dirs)
        }
        self.climate_paths = [os.path.join(root_dir, county_name, f"Raw/Climate/{yr}/DayMet_{county_name}_{yr}.nc") for yr in self.year]
        # self.soil_dir = os.path.join(root_dir, county_name, "Raw/Soil/spatial/soilmu_a_*.shp")
        import glob
        self.soil_dir = glob.glob(os.path.join(root_dir, county_name, "Raw/Soil/spatial/soilmu_a_*.shp"))

        if self.soil_dir:
            self.soil_dir = self.soil_dir[0]  # Take the first matching file
        else:
            print("No shapefile found.")

    def __call__(self, output_type: str | List[str] = "all", daily_climate: bool = True):
        """
        Args:
            output_type (str or list[str]): Specify the desired output(s).
                - "all" for all outputs (default)
                - Any combination of: "landsat_data", "et_data", "climate_data", "soil_data"
            daily_climate (bool): Whether to use daily climate data.

        Returns:
            dict or the requested subset of outputs.
        """
        # Ensure output_type is a list for consistent processing
        if output_type == "all":
            requested_outputs = ["landsat_data", "et_data", "climate_data", "soil_data"]
        elif isinstance(output_type, str):
            requested_outputs = [output_type]
        elif isinstance(output_type, list):
            requested_outputs = output_type
        else:
            raise ValueError("Invalid output_type. Must be 'all', a string, or a list of strings.")

        # Validate requested outputs
        valid_outputs = {"landsat_data", "et_data", "climate_data", "soil_data"}
        invalid_outputs = [key for key in requested_outputs if key not in valid_outputs]
        if invalid_outputs:
            raise ValueError(f"Invalid output(s) requested: {invalid_outputs}. Valid options are {valid_outputs}.")

        # Initialize cultivated area only once if needed
        cdl_cultivated_data = None
        if any(key in requested_outputs for key in valid_outputs):
            cdl_cultivated_data = self.get_cultivated_area(crop_names=self.crop_names)

        # Compute requested outputs
        outputs = {}
        if "landsat_data" in requested_outputs:
            outputs["landsat_data"] = self.get_masked_landsat_timeseries(cdl_cultivated_data)
        if "et_data" in requested_outputs:
            outputs["et_data"] = self.get_masked_et_timeseries(cdl_cultivated_data)
        if "climate_data" in requested_outputs:
            outputs["climate_data"] = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
        if "soil_data" in requested_outputs:
            outputs["soil_data"] = self.get_soil_dataset(cdl_cultivated_data)

        # Return requested outputs
        if len(requested_outputs) == 1:
            return outputs[requested_outputs[0]]  # Return single output directly
        return outputs
        

    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        bounds = self.dataframe[self.dataframe["NAME"] == self.county_name + " County"].total_bounds
        resolution = (30, 30)  # 30m resolution as Landsat uses
        return bounds, self.target_crs, resolution

    def align_to_geometry(self, reference_bounds, reference_crs, resolution, source_data, source_transform, source_crs):
        """
        Align source data to match the CRS, resolution, and bounds of the reference grid.

        Args:
            reference_bounds (tuple): Bounds of the reference grid (min_x, min_y, max_x, max_y).
            reference_crs: CRS of the reference grid.
            resolution (tuple): Resolution of the reference grid (x_res, y_res).
            source_data: Data to be aligned (numpy array).
            source_transform: Transform of the source data.
            source_crs: CRS of the source data.

        Returns:
            numpy.ndarray: Aligned data with the same resolution and extent as the reference.
        """
        min_x, min_y, max_x, max_y = reference_bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        reference_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        aligned_data = numpy.zeros((source_data.shape[0], height, width), dtype=source_data.dtype)

        for band_index in range(source_data.shape[0]):
            reproject(
                source=source_data[band_index],
                destination=aligned_data[band_index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear
            )

        return aligned_data, reference_transform, (height, width)

    def get_cultivated_area(self, crop_names=None):
        """
        Extract cultivated area and align it to the reference grid created from the county geometry.
        """
        with rasterio.open(self.cdl_paths[0]) as cdl_src:
            cdl_data = cdl_src.read(1)
            cdl_transform = cdl_src.transform
            cdl_crs = cdl_src.crs

            # Select crop codes
            if crop_names:
                if not isinstance(crop_names, list):
                    crop_names = [crop_names]
                crop_codes = [code for code, name in CDL_CROP_LEGEND.items() if name in crop_names]
                if not crop_codes:
                    raise ValueError("None of the specified crop names are valid.")
            else:
                crop_codes = [
                    code for code in CDL_CROP_LEGEND
                    if CDL_CROP_LEGEND[code] not in ["Non-agricultural", "NLCD-sampled categories", "Other"]
                ]

            mask = numpy.isin(cdl_data, crop_codes)
            cultivated_area = numpy.where(mask, cdl_data, 0)

            bounds, crs, resolution = self.get_reference_grid()
            aligned_cultivated_area, _, _ = self.align_to_geometry(
                reference_bounds=bounds,
                reference_crs=crs,
                resolution=resolution,
                source_data=numpy.expand_dims(cultivated_area, axis=0),
                source_transform=cdl_transform,
                source_crs=cdl_crs
            )

        return aligned_cultivated_area

    def get_masked_landsat_timeseries(self, cultivated_area):
        """
        Mask Landsat timeseries imagery using the cultivated area and stack into a 4D matrix.
        """
        timeseries = []
        bounds, crs, resolution = self.get_reference_grid()

        for landsat_path in self.landsat_files[self.year[0]]:
            with rasterio.open(landsat_path) as src:
                landsat_data = src.read()
                landsat_transform = src.transform
                landsat_crs = src.crs

                aligned_landsat, _, _ = self.align_to_geometry(
                    reference_bounds=bounds,
                    reference_crs=crs,
                    resolution=resolution,
                    source_data=landsat_data,
                    source_transform=landsat_transform,
                    source_crs=landsat_crs
                )

                mask = cultivated_area > 0
                mask = numpy.expand_dims(mask, axis=0)
                masked_landsat = aligned_landsat * mask
                timeseries.append(masked_landsat)

        return numpy.stack(timeseries, axis=0)

    def get_masked_et_timeseries(self, cultivated_area):
        """
        Mask ET timeseries imagery using the cultivated area and stack into a 4D matrix.
        """
        timeseries = []
        bounds, crs, resolution = self.get_reference_grid()

        for et_path in self.et_files[self.year[0]]:
            with rasterio.open(et_path) as src:
                et_data = src.read()
                et_transform = src.transform
                et_crs = src.crs

                aligned_et_data, _, _ = self.align_to_geometry(
                    reference_bounds=bounds,
                    reference_crs=crs,
                    resolution=resolution,
                    source_data=et_data,
                    source_transform=et_transform,
                    source_crs=et_crs
                )

                mask = cultivated_area > 0
                mask = numpy.expand_dims(mask, axis=0)
                masked_et = aligned_et_data * mask
                timeseries.append(masked_et)

        return numpy.stack(timeseries, axis=0)

    def get_soil_dataset(self, cultivated_area):
        """
        Rasterize SSURGO shapefile to match the reference grid and mask it with cultivated area.
        """
        gdf = geopandas.read_file(self.soil_dir)
        bounds, crs, resolution = self.get_reference_grid()

        gdf = gdf.to_crs(crs)  # Reproject shapefile to match reference CRS
        gdf["MUSYM_CODE"] = gdf["MUSYM"].astype("category").cat.codes

        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf["MUSYM_CODE"])]

        min_x, min_y, max_x, max_y = bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            dtype="int32"
        )

        masked_raster = numpy.where(cultivated_area[0] > 0, raster, 0)
        return masked_raster

    def get_climate_stack(self, cultivated_area, daily: bool = True):
        """
        Processes climate data from NetCDF files, resamples to 1km resolution, and masks with cultivated area.

        Args:
            cultivated_area (numpy.ndarray): Mask for the cultivated area.
            daily (bool): If True, return daily observations (364 times). If False, return monthly means (12 times).

        Returns:
            numpy.ndarray: Climate data stack in format (T, C, H, W).
        """

        variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'vp', 'snow', 'pet']
        timeseries = []
        bounds, crs, resolution = self.get_reference_grid()

        for climate_path in self.climate_paths:
            climate_data = xarray.open_dataset(climate_path)[variables]
            daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            climate_data = climate_data.rio.write_crs(daymet_crs)

            climate_data = climate_data.rio.reproject(crs, resolution=(1000, 1000))
            climate_bounds = climate_data.rio.bounds()
            # aligned_bounds = (591345.0, 3962475.0, 751815.0, 4086765.0)  # From previous calculation
            climate_data = climate_data.rio.clip_box(*climate_bounds)

            target_shape = (climate_data.rio.height, climate_data.rio.width)

            resampled_cultivated_area = self.resample_cultivated_area(cultivated_area[0, ...], target_shape)

            if not daily:
                climate_data = climate_data.groupby("time.month").mean("time")
                time_steps = climate_data.month
            else:
                time_steps = climate_data.time

            for time_step in time_steps:
                time_slice = climate_data.sel(time=time_step) if daily else climate_data.sel(month=time_step)

                climate_stack = numpy.stack(
                    [time_slice[var].values for var in variables],
                    axis=0
                )

                mask = resampled_cultivated_area > 0
                mask = numpy.expand_dims(mask, axis=0)
                masked_climate_stack = climate_stack * mask

                timeseries.append(masked_climate_stack)

        return numpy.stack(timeseries, axis=0)

    def resample_cultivated_area(self, cultivated_area, target_shape):
        """
        Resample the cultivated_area to match the target shape.
        Ensures that any pixel in the resampled image is 1 if any of the higher-resolution
        contributing pixels was 1.

        Args:
            cultivated_area (numpy.ndarray): Binary mask for cultivated area (H, W).
            target_shape (tuple): Desired shape (H, W).

        Returns:
            numpy.ndarray: Resampled binary mask.
        """

        # Ensure the inumpyut is binary
        if not numpy.array_equal(cultivated_area, cultivated_area.astype(bool)):
            # raise ValueError("Inumpyut cultivated_area must already be binary.")
            cultivated_area = (cultivated_area > 0).astype(numpy.uint8)

        # Resample using maximum aggregation for binary values
        resampled_area = resize(
            cultivated_area.astype(numpy.float32),  # Use float32 for interpolation
            output_shape=target_shape,
            order=1,  # Bilinear interpolation to preserve the contribution of higher-resolution pixels
            anti_aliasing=False,
            preserve_range=True,
        )

        # Threshold: Any value > 0 means at least one contributing pixel was 1
        return (resampled_area > 0).astype(numpy.uint8)


class MosaicClip:
    def __init__(self, file_dir, county_names):
        self.file_dir = file_dir
        self.county_names = county_names

        self.target_crs = "EPSG:32610"  # UTM Zone 10N
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        # self.dataframe = self.dataframe.to_crs(epsg=int(self.target_crs.split(':')[-1]))

    def align_to_geometry(self, reference_bounds, reference_crs, resolution, source_data, source_transform, source_crs):
        """ Align source data to match reference grid CRS, resolution, and bounds. """
        min_x, min_y, max_x, max_y = reference_bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        reference_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Preallocate array for efficiency
        aligned_data = numpy.empty((source_data.shape[0], height, width), dtype=source_data.dtype)

        for band_index in range(source_data.shape[0]):
            reproject(
                source=source_data[band_index],
                destination=aligned_data[band_index],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear
            )

        return aligned_data, reference_transform, (height, width)
    
    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        bounds = self.dataframe[self.dataframe["NAME"] == self.county_name + " County"].total_bounds
        resolution = (30, 30)  # 30m resolution as Landsat uses
        return bounds, resolution
    
    def create_mosaic(self):
        # 1. Get list of Landsat files
        tif_files = glob.glob(os.path.join(self.file_dir, "LT_2022-07-01-*.tif"))
        if len(tif_files) == 0:
            raise ValueError("No matching .tif files found.")

        # 2. Extract date from first filename
        first_filename = os.path.basename(tif_files[0])
        date_str = first_filename.split("_")[1]  # '2022-07-01'
        try:
            date_formatted = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        except:
            date_formatted = "unknownDate"

        # 3. Open all TIFF files
        src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

        # 4. Mosaic all files
        mosaic, out_trans = merge(src_files_to_mosaic, resampling=Resampling.nearest)
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })


        return mosaic, out_meta

    def run(self):

        mosaic, out_meta = self.create_mosaic()

        # # 6. Loop through counties
        for county in self.county_names:
    
            # geometry, polygon, ee_geometry, county_name_modified = get_county_info(df, county)
            normalized_county_name = county.strip().title() + " County"
            county_row = self.dataframe[self.dataframe["NAME"] == normalized_county_name]
            # county_row = df[df['NAME'].str.lower() == county.lower()]
            if county_row.empty:
                print(f"County '{county}' not found in shapefile.")
                continue

            geometry = county_row.geometry.values[0]
            print()
            # Clip the mosaic with the county geometry
            try:
                clipped, clipped_transform = mask(dataset = mosaic, shapes=[geometry], crop=True)
            except Exception as e:
                print(f"Failed to mask for county {county}: {e}")
                continue

            # Update metadata for clipped image
            clipped_meta = out_meta.copy()
            clipped_meta.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform
            })

            # Save the clipped image
            # county_clean_name = county.replace(" ", "_")
            output_filename = f"{normalized_county_name}.tif"
            output_path = os.path.join(self.file_dir, output_filename)

            with rasterio.open(output_path, "w", **clipped_meta) as dest:
                dest.write(clipped)

            print(f"Saved: {output_path}")

#***********************************************#
#****************** Founctions *****************#
#***********************************************#
def stratified_sampling(array, num_samples=128):
    """
    Apply stratified sampling along the last dimension of an array with shape (..., N).
    
    Parameters:
    - array (np.ndarray): Input array of shape (..., N)
    - num_samples (int): Number of samples to return along the last dimension
    
    Returns:
    - np.ndarray: Sampled array with shape (..., num_samples)
    """
    percentiles = numpy.linspace(0, 100, num_samples)

    if array.ndim == 2:
        return numpy.percentile(array, percentiles, axis=-1).swapaxes(-1, -2)
    if array.ndim == 3:
        T, B, N = array.shape
        return numpy.percentile(array, percentiles, axis=-1).transpose(1, 2, 0)  # Ensure (T, B, 128)
    
def modify_soil_attr_df(county_name: str = 'Yolo'):

    root_path = f'/data2/hkaman/Data/YieldBenchmark/counties/{county_name}/Raw/Soil/tabular'
    # root_path = '/data2/hkaman/Data/FoundationModel/CASoilData/tabular'
    df_path = os.path.join(root_path, 'muaggatt.txt')
    
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"File not found: {df_path}")

    # Read file without header first to check column count
    df = pandas.read_csv(df_path, delimiter="|", header=None)

    coln_list = [
        'musym', 'muname', 'mustatus', 'slopegraddcp', 'slopegradwta', 
        'brockdepmin', 'wtdepannmin', 'wtdepaprjunmin', 'flodfreqdcd', 
        'flodfreqmax', 'pondfreqprs', 'aws025wta', 'aws050wta', 'aws0100wta', 
        'aws0150wta', 'drclassdcd', 'drclasswettest', 'hydgrpdcd', 'iccdcd', 'iccdcdpct', 
        'niccdcd', 'niccdcdpct', 'engdwobdcd', 'engdwbdcd', 'engdwbll', 'engdwbml', 'engstafdcd', 
        'engstafll', 'engstafml', 'engsldcd', 'engsldcp', 'englrsdcd', 'engcmssdcd', 'engcmssmp', 
        'urbrecptdcd', 'urbrecptwta', 'forpehrtdcp', 'hydclprs', 'awmmfpwwta', 'mukey'
    ]

    # Ensure the number of columns matches
    if len(df.columns) == len(coln_list):
        df.columns = coln_list
    else:
        raise ValueError("The number of columns in the file does not match the expected column names.")


    # List of relevant soil attributes
    soil_attributes = [
        "aws0100wta", "slopegraddcp", "awmmfpwwta", "drclassdcd", "hydgrpdcd", "mukey", "cokey"
    ]

    # Select only available columns
    available_columns = [col for col in soil_attributes if col in df.columns]
    df_new = df[available_columns].copy()

    # Fill NaN values using the closest row first (bfill + ffill)
    df_new.fillna(method='bfill', inplace=True)  # Fill with next valid value
    df_new.fillna(method='ffill', inplace=True)  # Fill with previous valid value

    # Fill any remaining NaNs using random sampling from the same column
    for col in df_new.columns:
        if df_new[col].isna().sum() > 0:  # Check if NaNs are still present
            non_nan_values = df_new[col].dropna().values
            if len(non_nan_values) > 0:
                df_new[col] = df_new[col].apply(lambda x: numpy.random.choice(non_nan_values) if pandas.isna(x) else x)

    df_new.to_csv(os.path.join(root_path, 'muaggatt.csv'))
    return df_new

def get_county_info(dataframe, county_name):
    """
    Processes a county name to find the corresponding county in the DataFrame,
    retrieves its geometry, and prepares it for Earth Engine operations.

    Args:
        county_name (str): The name of the county (case insensitive, without "County").

    Returns:
        tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
    """
    # Normalize the inumpyut county name to match the DataFrame format
    normalized_county_name = county_name.strip().title() + " County"

    # Find the index of the matching county
    try:
        county_index = dataframe[dataframe["NAME"] == normalized_county_name].index[0]
    except IndexError:
        print(f"County '{county_name}' not found in the DataFrame.")
        return

    # Retrieve the geometry of the county
    polygon = dataframe.iloc[county_index].geometry
    aoi_geojson = geojson.Feature(geometry=mapping(polygon))
    geometry = aoi_geojson["geometry"]

    # Prepare the geometry for Earth Engine
    try:
        ee_geometry = get_flexible_geometry(geometry)
    except ValueError as e:
        print(f"Error processing geometry for county '{county_name}': {e}")
        return

    county_name_df = dataframe.iloc[county_index]["NAME"]
    county_name_modified = county_name_modification(county_name_df)

    return geometry, polygon, ee_geometry, county_name_modified

def get_county_geometry(self, county_name: str):
    """
    Retrieve the geometry of a specified county.

    Args:
        county_name (str): The name of the county (case insensitive, without "County").

    Returns:
        dict: Geometry of the county in GeoJSON format.
    """
    normalized_county_name = county_name.strip().title() + " County"

    try:
        county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
    except IndexError:
        raise ValueError(f"County '{county_name}' not found in the DataFrame.")

    polygon = self.dataframe.iloc[county_index].geometry
    return mapping(polygon)

def get_start_end_year_dates(year: int):

    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'  

    return start_date, end_date

def county_name_modification(county_name: str) -> str:

    if county_name.endswith(" County"):
        county_name = county_name[:-7]
    county_name = county_name.replace(" ", "")

    return county_name

def get_flexible_geometry(geometry):
    """
    Convert a GeoPandas geometry to an Earth Engine-compatible geometry.

    Args:
        geometry (shapely.geometry): The inumpyut geometry (Polygon, MultiPolygon, etc.).

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

def get_monthly_dates(year, index):
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
    last_day = calendar.monthrange(year, month)[1]
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-{last_day}'

    
    return start_date, end_date
    
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
            array (numpy.ndarray): Inumpyut array to normalize.
        
        Returns:
            numpy.ndarray: Normalized array.
        """
        nan_mask = numpy.isnan(array)
        if nan_mask.any():
            array[nan_mask] = numpy.nanmin(array)
        clip_percentile=(2, 98)
        p_low, p_high = numpy.percentile(array, clip_percentile)    
        return (array - p_low) / (p_high - p_low)
    
    # def normalize(array, clip_percentile=(2, 98)):
    #     """
    #     Normalize a NumPy array to [0, 1] range for brighter visualization,
    #     with optional percentile-based contrast stretching.

    #     Args:
    #         array (numpy.ndarray): Input array to normalize.
    #         clip_percentile (tuple): Percentiles for clipping (e.g., (2, 98)).

    #     Returns:
    #         numpy.ndarray: Normalized array for display.
    #     """
    #     nan_mask = numpy.isnan(array)
    #     if nan_mask.any():
    #         array[nan_mask] = numpy.nanmin(array)

    #     # Contrast stretching
    #     p_low, p_high = numpy.percentile(array, clip_percentile)
    #     array = numpy.clip(array, p_low, p_high)

    #     # Normalize to [0, 1]
    #     array = (array - p_low) / (p_high - p_low)
    #     return array
    
    # Open the Landsat file
    with rasterio.open(tif_file_path) as src:
        # Read bands
        red_band = src.read(3)  # Red is band 3 in Landsat 5
        green_band = src.read(2)  # Green is band 2
        blue_band = src.read(1)  # Blue is band 1

        # # Print original shape
        # print(f"Original shape: {red_band.shape}")
        # print(f"Original CRS: {src.crs}")

        # # Target CRS
        # target_crs = "EPSG:3857"

        # # Calculate transform and new shape for target CRS
        # transform, width, height = calculate_default_transform(
        #     src.crs, target_crs, src.width, src.height, *src.bounds
        # )

        # Prepare arrays for reprojected bands
        # red_reprojected = numpy.empty((height, width), dtype=red_band.dtype)
        # green_reprojected = numpy.empty((height, width), dtype=green_band.dtype)
        # blue_reprojected = numpy.empty((height, width), dtype=blue_band.dtype)

        # # Reproject each band
        # reproject(
        #     source=red_band,
        #     destination=red_reprojected,
        #     src_transform=src.transform,
        #     src_crs=src.crs,
        #     dst_transform=transform,
        #     dst_crs=target_crs,
        #     resampling=Resampling.nearest
        # )
        # reproject(
        #     source=green_band,
        #     destination=green_reprojected,
        #     src_transform=src.transform,
        #     src_crs=src.crs,
        #     dst_transform=transform,
        #     dst_crs=target_crs,
        #     resampling=Resampling.nearest
        # )
        # reproject(
        #     source=blue_band,
        #     destination=blue_reprojected,
        #     src_transform=src.transform,
        #     src_crs=src.crs,
        #     dst_transform=transform,
        #     dst_crs=target_crs,
        #     resampling=Resampling.nearest
        # )

        # # Print reprojected shape
        # print(f"Reprojected shape: {red_reprojected.shape}")
        # print(f"Reprojected CRS: {target_crs}")

    # Normalize bands for visualization
    red_band = normalize(red_band)
    green_band = normalize(green_band)
    blue_band = normalize(blue_band)

    # Stack bands into RGB
    rgb = numpy.stack([red_band, green_band, blue_band], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')  # Set figure (outside) background
    ax.set_facecolor('white')         # Set axes (plot area) background
    ax.imshow(rgb)
    ax.axis('off')
    ax.set_title("Reprojected Landsat 5 RGB Image (EPSG:3857)")
    plt.tight_layout()
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
    return xarray.open_dataset(file_path, engine="rasterio")#.to_array()

def plot_landsat5_all_bands_by_index(folder_path, index):
    """
    Plot all six Landsat 5 bands (RGB, NIR, SWIR1, SWIR2) separately in a 2-row by 3-column layout.

    Args:
        folder_path (str): Path to the folder containing .tif files.
        index (int): Index of the file to read.

    Returns:
        None. Displays the 6-band images.
    """
    def normalize(array):
        """
        Normalize the array to [0, 1] for better visualization after handling NaN values.
        """
        nan_mask = numpy.isnan(array)
        if nan_mask.any():
            array[nan_mask] = numpy.nanmin(array)

        norm_array = (array - array.min()) / (array.max() - array.min())

        # Set NaN values (previously no-data regions) to 1 (white)
        norm_array[nan_mask] = 1.0

        return norm_array

    # Open the .tif file with rioxarray
    image_xr = read_tif_by_index(folder_path, index)
    image_xr = image_xr.to_array()

    # Extract bands: Landsat 5 band order is assumed as:
    # 1: Blue, 2: Green, 3: Red, 4: NIR, 5: SWIR1, 7: SWIR2
    band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    bands = [normalize(image_xr.isel(band=i).values) for i in range(6)]

    # Create figure with a white background
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor="white")

    for ax, band, name in zip(axes.flat, bands, band_names):
        ax.set_facecolor("white")  # Set individual subplot background to white
        im = ax.imshow(band[0], cmap="gray", vmin=0, vmax=1)  # Plot the band
        ax.set_title(name, fontsize=12, color="black")  # Set title color to black
        ax.axis("off")  # Hide axes

    plt.suptitle("Landsat 5 Individual Bands", fontsize=14, fontweight="bold", color="black")
    plt.tight_layout()
    plt.show()

def plot_landsat5_rgb_by_index(folder_path, index):
    """
    Plot an RGB image from a Landsat 5 surface reflectance .tif file using rioxarray.

    Args:
        folder_path (str): Path to the folder containing .tif files.
        index (int): Index of the file to read.

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
        nan_mask = numpy.isnan(array)
        if nan_mask.any():
            array[nan_mask] = numpy.nanmin(array)

        # Normalize the array
        return (array - array.min()) / (array.max() - array.min())

    # Open the .tif file with rioxarray
    image_xr = read_tif_by_index(folder_path, index)
    image_xr = image_xr.to_array()

    # Select the Red, Green, and Blue bands (3, 2, 1 for Landsat 5)
    red_band = image_xr.isel(band=4)  # Band index is 0-based
    green_band = image_xr.isel(band=3)
    blue_band = image_xr.isel(band=2)

    # Normalize each band
    red = normalize(red_band.values)
    green = normalize(green_band.values)
    blue = normalize(blue_band.values)

    # Stack the bands into an RGB image
    rgb = numpy.stack([red, green, blue], axis=-1)

    # Convert NaNs to white (1,1,1)
    rgb = numpy.nan_to_num(rgb, nan=1.0)  # Set NaN values to white (RGB: 1,1,1)

    # Plot the RGB image
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('white')  # Set background color to white
    im = ax.imshow(rgb[0], vmin=0, vmax=1)  # Ensure correct scaling
    ax.axis('off')  # Remove axis for better visualization
    plt.title("Landsat 5 RGB Image", fontsize=14)
    plt.show()

def detect_tif_size_outliers(directory, tolerance_mb=10):
    """
    Detect .tif files in the directory that differ significantly in size from the others.

    Parameters:
    - directory (str): Path to the folder containing .tif files
    - tolerance_mb (int): Size deviation threshold in megabytes (default: 10 MB)
    """

    for year_folder in sorted(os.listdir(directory)):
        year_path = os.path.join(directory, year_folder)

        tif_files = [f for f in sorted(os.listdir(year_path)) if f.lower().endswith('.tif')]
        sizes = []

        for file in tif_files:
            file_path = os.path.join(year_path, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB
            sizes.append((file, size_mb))

        if not sizes:
            print("No .tif files found.")
            return

        # Compute median size as baseline
        size_values = [s for _, s in sizes]
        median_size = sorted(size_values)[len(size_values) // 2]

        print(f"Median file size: {median_size:.2f} MB")
        print(f"Files with size difference > {tolerance_mb} MB:")

        outliers = []
        for file, size in sizes:
            if abs(size - median_size) > tolerance_mb:
                outliers.append((file, size))

        if outliers:
            for file, size in outliers:
                print(f"  - {file} ({size:.2f} MB)")
        else:
            print("  None 🚀")
            
def count_observations_by_month(parent_folder, county_name: str):
    """
    Counts the number of observations (TIF files) per month for each year across subfolders
    and outputs a CSV file.

    Args:
        parent_folder (str): Path to the parent folder (e.g., YOLO).

    Returns:
        pd.DataFrame: A DataFrame with years as rows, months as columns, and counts of observations.
    """
    observations = defaultdict(int)

    for year_folder in sorted(os.listdir(parent_folder)):
        year_path = os.path.join(parent_folder, year_folder)

        if os.path.isdir(year_path) and year_folder.isdigit() and len(year_folder) == 4:
            year = int(year_folder)

            for file in os.listdir(year_path):
                if file.endswith(".tif"):
                    parts = file[:-4].split("_")  # remove .tif and split by "_"
                    if len(parts) == 4:
                        try:
                            file_year = int(parts[2])
                            month = int(parts[3])
                            if file_year == year:
                                observations[(year, month)] += 1
                        except ValueError:
                            print(f"Skipping file due to invalid date parts: {file}")

    # Create DataFrame
    years = sorted(set(year for year, _ in observations.keys()))
    months = [f"{month:02}" for month in range(1, 13)]
    data = {
        month: [observations.get((year, int(month)), 0) for year in years]
        for month in months
    }
    df = pandas.DataFrame(data, index=years)
    df.index.name = "Year"

    # Save to CSV
    output_csv = os.path.join(parent_folder, f"{county_name}_opm.csv")
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
                    dataset = xarray.open_dataarray(file_path)

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
        reprojected_data = numpy.empty((height, width), dtype=et_data.dtype)
        
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

def safe_read_csv(file_path, key_names, prefix):
    """Reads a CSV file safely, assigns unique prefixes to avoid merge conflicts, and extracts key columns."""
    try:
        df = pandas.read_csv(file_path, delimiter="|", header=None, low_memory=False)

        # # Assign generic column names
        # df.columns = [f"{prefix}_Column_{i}" for i in range(df.shape[1])]

        # # Rename last two columns as keys (MUKEY, COKEY, CHKEY)
        # key_mapping = {df.columns[-(i+1)]: key for i, key in enumerate(reversed(key_names))}
        # df.rename(columns=key_mapping, inumpylace=True)

        return df
    

    except (FileNotFoundError, pandas.errors.EmptyDataError):
        print(f"Warning: {file_path} is missing or empty.")
        return pandas.DataFrame()

def plot_datasets(landsat_data, et_data, climate_data, soil_data):

    # Ensure Landsat has at least 3 bands for RGB
    if landsat_data.shape[0] < 3:
        raise ValueError("Landsat data must have at least 3 bands for RGB plotting.")
        # Resample climate data to match Landsat resolution
    def resample_to_match(target_shape, source_data):
        """
        Resample source data to match the target shape using bilinear interpolation.
        """
        zoom_factors = (target_shape[0] / source_data.shape[0], target_shape[1] / source_data.shape[1])
        return zoom(source_data, zoom_factors, order=1)  # Bilinear interpolation


    # # Normalize Landsat RGB bands
    def normalize(array, brightness=1.0):
        """
        Normalize array using 5th and 95th percentiles for visualization.
        NaN values are replaced with 0.
        """
        min = numpy.nanmin(array)
        max = numpy.nanmax(array)

        array = numpy.nan_to_num(array, nan=0)  # Replace NaN with 0
        # p5, p95 = numpy.percentile(array, [5, 95])  # Compute 5th and 95th percentiles
        # print(p5, p95)
        min = numpy.nanmin(array)
        max = numpy.nanmax(array)

        # array = numpy.clip(array, p5, p95)  # Clip values to the percentile range
        normalized = (array - min) / (max - min + 1e-10)  # Normalize to [0, 1]
        return numpy.clip(normalized * brightness, 0, 1) 

    red = normalize(landsat_data[2], brightness=3)  
    green = normalize(landsat_data[1], brightness=3)  
    blue = normalize(landsat_data[0], brightness=3)  

    # Stack RGB bands into an RGB image
    rgb_image = numpy.stack([red, green, blue], axis=-1)



    # # Identify fully white background (NaN or 0 in all datasets)
    # full_white_background = numpy.all(numpy.isnan(landsat_data) | (landsat_data == 0), axis=0)

    # # Identify light gray background (valid values outside main region)
    # valid_mask = ~full_white_background
    # light_gray_background = (landsat_data.sum(axis=0) > 0) & (~valid_mask)

    # # Landsat visualization with light gray and white backgrounds
    # visualization_image = numpy.ones_like(rgb_image)  # Initialize with white background
    # visualization_image[valid_mask] = rgb_image[valid_mask]  # Apply Landsat data to valid areas
    # visualization_image[light_gray_background] = [0.95, 0.95, 0.95]  # Light gray for valid areas outside the main region

    # # Replace NaN or 0 in other datasets with white or light gray backgrounds
    # def apply_background(data):
    #     data_with_bg = numpy.copy(data)
    #     data_with_bg[full_white_background] = numpy.nan  # Fully white for full background
    #     data_with_bg[light_gray_background] = 0  # Light gray for secondary background
    #     return data_with_bg

    # et_data = apply_background(et_data)
    # climate_data_resampled = apply_background(climate_data_resampled)
    # soil_data = apply_background(soil_data)

    # # Create colormap for soil data
    # unique_soil_types = numpy.unique(soil_data[~numpy.isnan(soil_data)])  # Only non-NaN values
    # colormap_soil = plt.cm.get_cmap("tab20", len(unique_soil_types))
    # color_mapping_soil = {soil_type: colormap_soil(i) for i, soil_type in enumerate(unique_soil_types)}
    # cmap_soil = ListedColormap([color_mapping_soil[soil_type] for soil_type in unique_soil_types])

    # # Plot all datasets
    # fig, axes = plt.subplots(1, 4, figsize=(25, 5), constrained_layout=True)

    # datasets = [visualization_image, et_data, climate_data_resampled, soil_data]
    # titles = ["Landsat (RGB)", "ET", "Climate (Resampled)", "Soil"]
    # cmaps = [None, 'coolwarm', 'plasma', cmap_soil]

    # for ax, data, title, cmap in zip(axes, datasets, titles, cmaps):
    #     if cmap is None:  # For RGB image
    #         im = ax.imshow(data)
    #     else:
    #         im = ax.imshow(data, cmap=cmap)
    #     ax.set_title(title, fontsize=14)
    #     ax.axis('off')
    #     if cmap is not None:
    #         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # plt.show()
    # Create colormap for soil data
    unique_soil_types = numpy.unique(soil_data[soil_data > 0])
    colormap_soil = plt.cm.get_cmap("tab20", len(unique_soil_types))
    color_mapping_soil = {soil_type: colormap_soil(i) for i, soil_type in enumerate(unique_soil_types)}
    cmap_soil = ListedColormap([color_mapping_soil[soil_type] for soil_type in unique_soil_types])

    # Plot all datasets
    fig, axes = plt.subplots(1, 4, figsize=(25, 5), constrained_layout=True)

    datasets = [rgb_image, et_data, climate_data, soil_data]
    titles = ["Landsat", "ET", "Climate", "Soil"]
    cmaps = ['viridis', 'coolwarm', 'plasma', cmap_soil]

    for ax, data, title, cmap in zip(axes, datasets, titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

def plot_soil_vars(attribute_matrix):

    soil_attributes = [
    "aws0100wta", "slopegraddcp", "awmmfpwwta", "drclassdcd", "hydgrpdcd"
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 3), facecolor='white')

    # Plot each soil variable
    for i in range(5):
        ax = axes[i]
        im = ax.imshow(attribute_matrix[i], cmap='viridis', aspect='auto')
        ax.set_facecolor("white")
        ax.set_title(soil_attributes[i])
        ax.axis("off")  # Hide axes for better visualization
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def ca_soil_vars_gen():

    target_crs = "EPSG:32610"

    root_dir = '/data2/hkaman/Data/FoundationModel/Inputs'

    dataframe = geopandas.read_file('/data2/hkaman/Data/FoundationModel/SHPs/CASHP/CA_State.shp')
    dataframe = dataframe.to_crs(epsg=int(target_crs.split(':')[-1]))

    gdf = geopandas.read_file('/data2/hkaman/Data/FoundationModel/CASoilData/spatial/gsmsoilmu_a_ca.shp').to_crs(target_crs)

    # Get reference grid
    bounds = dataframe.total_bounds
    resolution = (30, 30)

    min_x, min_y, max_x, max_y = bounds
    width = int((max_x - min_x) / resolution[0])
    height = int((max_y - min_y) / resolution[1])
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Rasterize using MUKEY
    shapes = [(geom, int(mukey)) for geom, mukey in zip(gdf.geometry, gdf["MUKEY"])]
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, dtype=numpy.int32)

    # Load tabular data

    tabular_dir = '/data2/hkaman/Data/FoundationModel/CASoilData/tabular'
    muaggatt_df = pandas.read_csv(f"{tabular_dir}/muaggatt.csv")

    # Normalize column names
    muaggatt_df.columns = muaggatt_df.columns.str.lower().str.strip()

    # Soil attributes of interest

    soil_attributes = [
        "aws0100wta", "slopegraddcp", "awmmfpwwta", "drclassdcd", "hydgrpdcd"
    ]
    # Ensure "mukey" exists
    if "mukey" not in muaggatt_df.columns:
        raise ValueError("MUKEY column is missing in the CSV file.")

    # Convert categorical columns to numeric
    drainage_mapping = {
        "Excessively drained": 5.0, "Well drained": 4.0, "Moderately well drained": 3.0,
        "Somewhat poorly drained": 2.0, "Poorly drained": 1.0, "Very poorly drained": 0.0
    }
    soil_groups_mapping = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}

    if "drclassdcd" in muaggatt_df.columns:
        muaggatt_df["drclassdcd"] = muaggatt_df["drclassdcd"].map(drainage_mapping).astype(numpy.float32)
    if "hydgrpdcd" in muaggatt_df.columns:
        muaggatt_df["hydgrpdcd"] = muaggatt_df["hydgrpdcd"].map(soil_groups_mapping).astype(numpy.float32)

    # Convert mukey to int for efficient indexing
    muaggatt_df["mukey"] = muaggatt_df["mukey"].astype(int)
    existing_attributes = [attr for attr in soil_attributes if attr in muaggatt_df.columns]
    # Create lookup table for soil attributes
    lookup = muaggatt_df.set_index("mukey")[existing_attributes].to_dict(orient="index")

    # Convert raster to index-based lookup
    unique_mukeys = numpy.unique(raster)
    attribute_matrix = numpy.full((len(soil_attributes), height, width), numpy.nan, dtype=numpy.float32)

    for i, attr in enumerate(existing_attributes):
        attr_values = numpy.array([lookup.get(mukey, {}).get(attr, numpy.nan) for mukey in unique_mukeys])  # Default NaN instead of 0
        attr_map = numpy.full_like(raster, numpy.nan, dtype=numpy.float32)  # Initialize with NaN
        valid_pixels = raster > 0  # Identify valid raster pixels
        attr_map[valid_pixels] = numpy.take(attr_values, numpy.searchsorted(unique_mukeys, raster[valid_pixels]))
        attribute_matrix[i, :, :] = attr_map  # Store in matrix


    return attribute_matrix

def soil_vars_gen(county_name: str):

    target_crs = "EPSG:32610"

    root_dir = '/data2/hkaman/Data/FoundationModel/Inputs'

    dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
    dataframe = dataframe.to_crs(epsg=int(target_crs.split(':')[-1]))

    soil_dir = glob.glob(os.path.join(root_dir, county_name, "Raw/Soil/spatial/soilmu_a_*.shp"))
    soil_dir = soil_dir[0] 

    gdf = geopandas.read_file(soil_dir).to_crs(target_crs)

    # Get reference grid
    bounds = dataframe[dataframe["NAME"] == county_name + " County"].total_bounds
    resolution = (30, 30)

    min_x, min_y, max_x, max_y = bounds
    width = int((max_x - min_x) / resolution[0])
    height = int((max_y - min_y) / resolution[1])
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Rasterize using MUKEY
    shapes = [(geom, int(mukey)) for geom, mukey in zip(gdf.geometry, gdf["MUKEY"])]
    raster = rasterize(shapes, out_shape=(height, width), transform=transform, dtype=numpy.int32)

    # Load tabular data
    tabular_dir = soil_dir.rsplit("/", 1)[0].replace("spatial", "tabular")
    muaggatt_df = pandas.read_csv(f"{tabular_dir}/muaggatt.csv")

    # Normalize column names
    muaggatt_df.columns = muaggatt_df.columns.str.lower().str.strip()

    # Soil attributes of interest

    soil_attributes = [
        "aws0100wta", "slopegraddcp", "awmmfpwwta", "drclassdcd", "hydgrpdcd"
    ]
    # Ensure "mukey" exists
    if "mukey" not in muaggatt_df.columns:
        raise ValueError("MUKEY column is missing in the CSV file.")

    # Convert categorical columns to numeric
    drainage_mapping = {
        "Excessively drained": 5.0, "Well drained": 4.0, "Moderately well drained": 3.0,
        "Somewhat poorly drained": 2.0, "Poorly drained": 1.0, "Very poorly drained": 0.0
    }
    soil_groups_mapping = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0}

    if "drclassdcd" in muaggatt_df.columns:
        muaggatt_df["drclassdcd"] = muaggatt_df["drclassdcd"].map(drainage_mapping).astype(numpy.float32)
    if "hydgrpdcd" in muaggatt_df.columns:
        muaggatt_df["hydgrpdcd"] = muaggatt_df["hydgrpdcd"].map(soil_groups_mapping).astype(numpy.float32)

    # Convert mukey to int for efficient indexing
    muaggatt_df["mukey"] = muaggatt_df["mukey"].astype(int)
    existing_attributes = [attr for attr in soil_attributes if attr in muaggatt_df.columns]
    # Create lookup table for soil attributes
    lookup = muaggatt_df.set_index("mukey")[existing_attributes].to_dict(orient="index")

    # Convert raster to index-based lookup
    unique_mukeys = numpy.unique(raster)
    attribute_matrix = numpy.full((len(soil_attributes), height, width), numpy.nan, dtype=numpy.float32)

    for i, attr in enumerate(existing_attributes):
        attr_values = numpy.array([lookup.get(mukey, {}).get(attr, numpy.nan) for mukey in unique_mukeys])  # Default NaN instead of 0
        attr_map = numpy.full_like(raster, numpy.nan, dtype=numpy.float32)  # Initialize with NaN
        valid_pixels = raster > 0  # Identify valid raster pixels
        attr_map[valid_pixels] = numpy.take(attr_values, numpy.searchsorted(unique_mukeys, raster[valid_pixels]))
        attribute_matrix[i, :, :] = attr_map  # Store in matrix


    return attribute_matrix

def rename_folders_and_files(base_path):
    base_path = Path(base_path)

    # Loop through all counties
    for county_folder in base_path.iterdir():
        if county_folder.is_dir():
            # Step 1: Rename county folder if it contains "_"
            original_name = county_folder.name
            new_name = original_name.replace('_', '')
            new_county_path = base_path / new_name

            if original_name != new_name:
                print(f"Renaming county folder: {original_name} -> {new_name}")
                shutil.move(str(county_folder), str(new_county_path))
            else:
                new_county_path = county_folder

            # Step 2: Go into Raw/Landsat
            landsat_folder = new_county_path / "Raw" / "Landsat"
            if not landsat_folder.exists():
                print(f"Skipping missing Landsat folder: {landsat_folder}")
                continue

            # Step 3: Loop through year folders
            for year_folder in landsat_folder.iterdir():
                if year_folder.is_dir() and year_folder.name != '2012':
                    year = year_folder.name

                    for tif_file in year_folder.glob("*.tif"):
                        # Match any of these formats: LT_2008_01_01, LT_2008-01-01, County_2008_01_01
                        match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', tif_file.stem)
                        if match:
                            yr, month, day = match.groups()
                            new_filename = f"{new_name}_{yr}_{month}_01.tif"
                            new_file_path = tif_file.parent / new_filename

                            print(f"Renaming: {tif_file.name} -> {new_filename}")
                            tif_file.rename(new_file_path)
                        else:
                            print(f"Skipping file (unmatched format): {tif_file.name}")

def clean_crop_data(county_name:str):

    main_dir = os.path.join(SAVE_ROOT_DIR, f'counties/{county_name}/InD')
    # Define years (excluding 2012)
    years = [str(year) for year in range(2008, 2023) if year != 2012]
    # Define names to remove
    remove_names = {'Other Crops', 'Other Tree Crops', 'Other'}

    for year in years:
        year_folder = os.path.join(main_dir, year)
        if not os.path.isdir(year_folder):
            print(f"Skipping missing folder: {year_folder}")
            continue

        # Find CSV file in the year folder
        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file found in {year_folder}")
            continue
        csv_path = os.path.join(year_folder, csv_files[0])

        # Load and clean CSV
        df = pandas.read_csv(csv_path)
        if 'key_crop_name' not in df.columns:
            print(f"'key_crop_name' column not found in {csv_path}")
            continue

        df_cleaned = df[~df['key_crop_name'].isin(remove_names)]

        # Overwrite original CSV
        df_cleaned.to_csv(csv_path, index=False)
        print(f"Cleaned and saved: {csv_path}")

def aggregate_duplicate_crops(county_name:str):

    main_dir = os.path.join(SAVE_ROOT_DIR, f'counties/{county_name}/InD')


    years = [str(y) for y in range(2008, 2023) if y != 2012]

    for year in years:
        year_folder = os.path.join(main_dir, year)
        if not os.path.isdir(year_folder):
            print(f"Missing folder: {year_folder}")
            continue

        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file in {year_folder}")
            continue

        csv_path = os.path.join(year_folder, csv_files[0])
        df = pandas.read_csv(csv_path)

        if 'key_crop_name' not in df.columns:
            print(f"'key_crop_name' column missing in {csv_path}")
            continue

        # Check for duplicates
        if df['key_crop_name'].duplicated().any():
            # Aggregate
            aggregated_df = df.groupby('key_crop_name').agg({
                'harvested_acres': 'sum',
                'production': 'sum',
                'yield': 'mean',
                'crop_name': lambda x: list(x.unique()),
                'county': 'first',
                'year': 'first'
            }).reset_index()

            # Save the new aggregated CSV
            aggregated_df.to_csv(csv_path, index=False)
            print(f"Aggregated and saved: {csv_path}")
        else:
            print(f"No duplicates to aggregate in: {csv_path}")

def count_samples_by_crop_and_year(county_name:str):

    main_dir = os.path.join(SAVE_ROOT_DIR, f'counties/{county_name}/InD')

    years = [str(y) for y in range(2008, 2023) if y != 2012]
    crop_year_counts = {}

    for year in years:
        year_folder = os.path.join(main_dir, year)
        if not os.path.isdir(year_folder):
            print(f"Skipping missing folder: {year_folder}")
            continue

        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file in {year_folder}")
            continue

        csv_path = os.path.join(year_folder, csv_files[0])
        df = pandas.read_csv(csv_path)

        if 'key_crop_name' not in df.columns:
            print(f"'key_crop_name' column missing in {csv_path}")
            continue

        counts = df['key_crop_name'].value_counts()
        crop_year_counts[year] = counts

    # Combine all into a single DataFrame
    summary_df = pandas.DataFrame(crop_year_counts).fillna(0).astype(int)
    summary_df.index.name = 'key_crop_name'
    return summary_df

def remove_crops_missing_from_any_year(county_name:str):

    main_dir = os.path.join(SAVE_ROOT_DIR, f'counties/{county_name}/InD')

    years = [str(y) for y in range(2008, 2023) if y != 2012]
    crop_sets_by_year = {}

    # Step 1: Gather crop names present in each year
    for year in years:
        year_folder = os.path.join(main_dir, year)
        if not os.path.isdir(year_folder):
            print(f"Skipping: {year_folder}")
            continue

        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV in: {year_folder}")
            continue

        csv_path = os.path.join(year_folder, csv_files[0])
        df = pandas.read_csv(csv_path)

        if 'key_crop_name' not in df.columns:
            print(f"'key_crop_name' missing in: {csv_path}")
            continue

        crop_sets_by_year[year] = set(df['key_crop_name'].dropna().unique())

    # Step 2: Find crops that are common to *all* years
    common_crops = set.intersection(*crop_sets_by_year.values())

    # Step 3: Remove crops not in common_crops from each CSV and overwrite
    for year in years:
        year_folder = os.path.join(main_dir, year)
        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            continue

        csv_path = os.path.join(year_folder, csv_files[0])
        df = pandas.read_csv(csv_path)

        if 'key_crop_name' not in df.columns:
            continue

        cleaned_df = df[df['key_crop_name'].isin(common_crops)]
        cleaned_df.to_csv(csv_path, index=False)
        print(f"Cleaned {csv_path} to keep only crops in ALL years.")

    print(f"✅ Crops retained across all years: {sorted(common_crops)}")

def filter_csv_by_npz_keys(county_name:str):


    main_dir = os.path.join(SAVE_ROOT_DIR, f'counties/{county_name}/InD')
    years = [str(y) for y in range(2008, 2023) if y != 2012]

    for year in years:
        year_folder = os.path.join(main_dir, year)
        if not os.path.isdir(year_folder):
            print(f"Skipping missing folder: {year_folder}")
            continue
            
        # Get CSV file
        csv_files = [f for f in os.listdir(year_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file in {year_folder}")
            continue
        csv_path = os.path.join(year_folder, csv_files[0])

        # Get NPZ file
        npz_files = [f for f in os.listdir(year_folder) if f.endswith('.npz')]
        if not npz_files:
            print(f"No NPZ file in {year_folder}")
            continue
        npz_path = os.path.join(year_folder, npz_files[0])

        # Load CSV and NPZ
        df = pandas.read_csv(csv_path)

        npz_dict = dict(numpy.load(npz_path, allow_pickle=True)["inumpyut"].item())
        valid_keys = set(npz_dict.keys())

        # print(f"{year}:{df['key_crop_name'].unique()} | {valid_keys}")
        # Filter rows where key_crop_name IS in the dictionary keys
        if 'key_crop_name' not in df.columns:
            print(f"'key_crop_name' not found in {csv_path}")
            continue

        original_len = len(df)
        df_filtered = df[df['key_crop_name'].isin(valid_keys)]

        # Save the cleaned CSV
        df_filtered.to_csv(csv_path, index=False)
        print(f"[{year}] Kept {len(df_filtered)}/{original_len} rows in {csv_path}")

def plot_tiff_files(folder_path):
    # Get a list of all TIFF files in the folder
    tiff_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff') or f.endswith('.tif')]

    # Sort files for consistent order
    tiff_files.sort()

    # Month names for titles
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # Read all TIFF files and store data in a list
    data_list = []
    for tiff_file in tiff_files:
        with rasterio.open(tiff_file) as src:
            data = src.read(1)  # Read the first band
            # data = np.nan_to_num(data)
            data_list.append(data)

    # Determine the global min and max values across all files
    global_min = min([np.nanmin(data) for data in data_list])
    global_max = max([np.nanmax(data) for data in data_list])
    
    # Set up the plot grid
    fig, axes = plt.subplots(2, 6, figsize=(17, 6), constrained_layout=True)
    axes = axes.flatten()

    # Plot each TIFF file
    for i, (data, ax) in enumerate(zip(data_list, axes)):
        im = ax.imshow(data, cmap='viridis', vmin=global_min, vmax=global_max)
        ax.set_facecolor("white")
        ax.set_title(months[i], fontsize=10)
        ax.axis('off')

    # Remove any unused subplots
    for ax in axes[len(data_list):]:
        ax.axis('off')

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04, aspect=50)
    cbar.set_label('ET Value')

    plt.show()

def extract_formatted_date(filename):
    """
    Extracts and formats the date from a Landsat filename as 'YYYY_MM_DD'.
    
    Expected filename format: 'LT_YYYY-MM-DD-xxxxxxxxxx-xxxxxxxxxx.tif'
    
    Returns:
        A string like '2022_07_01'
    """
    base = os.path.basename(filename)
    
    # Use regex to find the date pattern
    match = re.search(r'LT_(\d{4})-(\d{2})-(\d{2})', base)
    if match:
        year, month, day = match.groups()
        return f"{year}_{month}_{day}"
    else:
        raise ValueError("Date not found or filename format is incorrect.")

def create_mosaic(dir):
    # 1. Get list of Landsat files
    tif_files = glob.glob(os.path.join(dir, "*.tif"))

    formatted_date = extract_formatted_date(tif_files[0])


    if len(tif_files) == 0:
        raise ValueError("No matching .tif files found.")

    # 2. Open all TIFF files
    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

    # 3. Mosaic all files
    mosaic, out_trans = merge(src_files_to_mosaic, resampling=Resampling.nearest)

    # 4. Force data to float32
    mosaic = mosaic.astype(numpy.float32)

    # 5. Update metadata
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw",
        "dtype": "float32"
    })

    return mosaic, out_meta, formatted_date

def reproject_and_plot_cdl(cdl_path, target_crs='EPSG:32610'):
    """
    Reproject a CDL raster to a target CRS and plot it using a provided color map.

    Parameters:
    - cdl_path (str): Path to the CDL raster file.
    - target_crs (str): Target coordinate reference system. Default is EPSG:32610.
    - cdl_colors (dict): Dictionary mapping CDL codes to RGB colors.
    """

    with rasterio.open(cdl_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        nodata_value = src.nodata

        transform, width, height = calculate_default_transform(
            src_crs, target_crs, src_width, src_height, *src.bounds
        )

        reprojected_data = np.empty((height, width), dtype=src.meta['dtype'])

        reproject(
            source=src.read(1),
            destination=reprojected_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

    cmap = ListedColormap([CDL_COLORS[key] for key in sorted(CDL_COLORS.keys())])
    cdl_data_masked = np.ma.masked_where(reprojected_data == nodata_value, reprojected_data)

    plt.figure(figsize=(12, 10))
    show(cdl_data_masked, cmap=cmap, transform=transform)
    plt.title("Reprojected CDL Raster")
    plt.axis('off')
    plt.show()

def clip_cdl_by_counties(year: str, county_names: list):
    """
    Clips a CDL GeoTIFF by a list of county geometries and saves each one.

    Parameters:
    - cdl_path: str, path to the CDL GeoTIFF file.
    - counties_gdf: GeoDataFrame, contains geometries and a column 'county_name'.
    - county_names: list of str, list of county names to clip and save.
    - output_dir: str, directory to save the clipped rasters.

    Returns:
    - None

    """ 

    dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
    cdl_path  = f'/data2/hkaman/Data/YieldBenchmark/CDLs/{year}/CA_CDL_{year}.TIF'
    with rasterio.open(cdl_path) as src:
        raster_crs = src.crs
        dataframe = dataframe.to_crs(raster_crs)

        for county in county_names:
            output_dir = f'/data2/hkaman/Data/YieldBenchmark/counties/{county}/Raw/CDL/{year}'
            os.makedirs(output_dir, exist_ok=True)


            normalized_county_name = county.strip().title() + " County"
            county_row = dataframe[dataframe["NAME"] == normalized_county_name]

            county_geom = county_row.geometry.values[0]
            try:
                out_image, out_transform = mask(
                    dataset = src,
                    shapes = [county_geom],
                    crop = True,
                    nodata = src.nodata
                )

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                out_path = os.path.join(output_dir, f"{county.replace(' ', '_')}_{year}.tif")
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"Saved {out_path}")
            except Exception as e:
                print(f"Failed to clip {county}: {e}")



def process_and_plot_daymet_climate(root_dir, county_name, yr, target_crs="EPSG:32610", daily=True):
    """
    Loads, reprojects, stacks, and plots DayMet climate data for a given county and year.

    Parameters:
    - root_dir (str): Root directory of the dataset.
    - county_name (str): Name of the county.
    - yr (int): Year of the data.
    - target_crs (str): CRS to project the data into. Default: EPSG:32610.
    - daily (bool): Whether to use daily timesteps or monthly averages. Default: True.

    Returns:
    - vector_out (np.ndarray): Array with shape (T, V, H, W) where T is time, V is variable count.
    """

    climate_path = os.path.join(root_dir, county_name, f"Raw/Climate/{yr}/DayMet_{county_name}_{yr}.nc")
    variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'vp', 'snow', 'pet']
    variable_names = variables.copy()

    # Load data and assign CRS
    daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    climate_data = xarray.open_dataset(climate_path)[variables].rio.write_crs(daymet_crs)

    # Reproject to target CRS and clip to bounds
    climate_data = climate_data.rio.reproject(target_crs, resolution=(1000, 1000))
    climate_data = climate_data.rio.clip_box(*climate_data.rio.bounds())

    # Convert time series to stacked format
    time_steps = climate_data.time
    vector_timeseries = []

    for time_step in time_steps:
        time_slice = climate_data.sel(time=time_step) if daily else climate_data.sel(month=time_step)
        climate_stack = numpy.stack(
            [time_slice[var].values for var in variables],
            axis=0
        ).astype(numpy.float32)

        vector_timeseries.append(climate_stack)

    vector_out = numpy.stack(vector_timeseries, axis=0)  # (T, V, H, W)

    # Plot a sample time step (e.g., 180th day)
    sample_index = 180 if vector_out.shape[0] > 180 else vector_out.shape[0] // 2
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i, var_name in enumerate(variable_names):
        row, col = divmod(i, 4)
        ax = axs[row, col]

        data = vector_out[sample_index, i, ...]
        im = ax.imshow(data, cmap='viridis')
        ax.set_facecolor("white")
        ax.set_title(var_name)
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

    return vector_out