import os
import numpy
import pandas
import geopandas
import xarray
import rioxarray
import geojson
import ee
import geemap
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import rasterize
from rasterio.transform import from_origin, from_bounds
from rasterio.mask import mask
import requests
from typing import List, Union
from datetime import date, timedelta
import zipfile
from pynhd import NLDI
from collections import defaultdict
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import wkt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
import pydaymet as daymet
import calendar
import json
from skimage.transform import resize
from rapidfuzz import process, fuzz  


# Defults Directries ; 

CA_COUNTIES_SHAPEFILE_DIR = '/data2/hkaman/Data/CDL/California_Counties.geojson'
DEFAULT_CRS = 4326
SAVE_ROOT_DIR = '/data2/hkaman/Data/FoundationModel'
YIELD_RAW_FILES_DIR = "/data2/hkaman/Data/FoundationModel/YieldObservation"
 

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
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg=DEFAULT_CRS)

    def __call__(self):
        return self.get_climate_data_county()
    

    def get_climate_data_county(self):
        _, geometry, _, county_name_modified = get_county_info(self.dataframe, self.county_name)

        dataset = self.get_daymet_data(geometry)

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
        self.dataframe = geopandas.read_file(CA_COUNTIES_SHAPEFILE_DIR)
        self.dataframe = self.dataframe.to_crs(epsg = DEFAULT_CRS)

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
        root_dir = '/data2/hkaman/Data/FoundationModel'
        self.output_dir = os.path.join(root_dir,county_name, f"InD/{self.year[0]}/{county_name}_{self.year[0]}.numpyz")

        # Paths for each year
        self.cdl_paths = [os.path.join(root_dir, county_name, f"CDL/{yr}/{county_name}_{yr}.tif") for yr in self.year]
        self.landsat_dirs = [os.path.join(root_dir, county_name, f"Landsat/{yr}/") for yr in self.year]
        self.landsat_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.landsat_dirs)
        }
        self.et_dirs = [os.path.join(root_dir, county_name, f"ET/{yr}/") for yr in self.year]
        self.et_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.et_dirs)
        }
        self.climate_paths = [os.path.join(root_dir, county_name, f"Climate/{yr}/DayMet_{county_name}_{yr}.nc") for yr in self.year]
        self.soil_dir = os.path.join(root_dir, county_name, "Soil/spatial/soilmu_a_ca053.shp")
    
    def __call__(self, output_type: str | List[str] = "all", daily_climate: bool = True):
        """
        Args:
            output_type (str or list[str]): Specify the desired output(s).
                - "all" for all outputs (default)
                - Any combination of: "landsat_data", "et_data", "climate_data", "soil_data"
            daily_climate (bool): Whether to use daily climate data.

        Returns:
            dict: A dictionary where each crop name maps to another dictionary containing requested datasets.
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

        # Initialize the output dictionary for multiple crops
        outputs = {}

        # Process each crop separately
        for crop_name in self.crop_names:
            # Get cultivated area for the current crop
            cdl_cultivated_data = self.get_cultivated_area(crop_name = crop_name)
            if cdl_cultivated_data is not None:

                crop_outputs = {}

                if "landsat_data" in requested_outputs:
                    landsat = self.get_masked_landsat_timeseries(cdl_cultivated_data)
                    crop_outputs["landsat_data"] = landsat
                if "et_data" in requested_outputs:
                    crop_outputs["et_data"] = self.get_masked_et_timeseries(cdl_cultivated_data)
                if "climate_data" in requested_outputs:
                    crop_outputs["climate_data"] = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)

                if "soil_data" in requested_outputs:
                    crop_outputs["soil_data"] = self.get_soil_dataset(cdl_cultivated_data)
                outputs[crop_name] = crop_outputs
            
        numpy.savez_compressed(self.output_dir , inumpyut = outputs)

        return outputs

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

    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        county_geometry = self.get_county_geometry(self.county_name)
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
                bounds, crs, resolution = self.get_reference_grid()
                aligned_cultivated_area, _, _ = self.align_to_geometry(
                    reference_bounds = bounds,
                    reference_crs = crs,
                    resolution = resolution,
                    source_data = numpy.expand_dims(cultivated_area, axis=0),
                    source_transform = cdl_transform,
                    source_crs = cdl_crs
                )

                return aligned_cultivated_area
    
    def get_masked_landsat_timeseries(self, cultivated_area):
        """
        Mask Landsat timeseries imagery using the cultivated area and return non-zero pixels as a 3D matrix (T, B, N).
        """
        timeseries = []
        bounds, crs, resolution = self.get_reference_grid()

        for landsat_path in self.landsat_files[self.year[0]]:
            with rasterio.open(landsat_path) as src:
                landsat_data = src.read()  # Shape: (Bands, H, W)
                landsat_transform = src.transform
                landsat_crs = src.crs

                # Align Landsat data to the reference grid
                aligned_landsat, _, _ = self.align_to_geometry(
                    reference_bounds=bounds,
                    reference_crs=crs,
                    resolution=resolution,
                    source_data=landsat_data,
                    source_transform=landsat_transform,
                    source_crs=landsat_crs
                )  # Output shape: (B, H, W)

                # Create mask and apply it
                mask = cultivated_area > 0  # Shape: (H, W)
                masked_landsat = aligned_landsat[:, mask[0, ...]]  # Output shape: (B, N) where N = non-zero pixels

                timeseries.append(masked_landsat)  # Append (B, N) for each timestamp

        # Stack along the time dimension
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

                mask = cultivated_area > 0  # Shape: (H, W)
                masked_et = aligned_et_data[:, mask[0, ...]]  # Output shape: (B, N) where N = non-zero pixels
                timeseries.append(masked_et)


        return numpy.stack(timeseries, axis=0) 
    
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

        # Get reference grid details (extent, CRS, resolution)
        bounds, crs, resolution = self.get_reference_grid()

        # Reproject to match reference grid CRS
        gdf = gdf.to_crs(crs)

        # Extract unique soil IDs (MUKEY)
        shapes = [(geom, int(value)) for geom, value in zip(gdf.geometry, gdf["MUKEY"])]

        # Define raster resolution
        min_x, min_y, max_x, max_y = bounds
        width = int((max_x - min_x) / resolution[0])
        height = int((max_y - min_y) / resolution[1])
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

        # Rasterize soil polygons using MUKEY
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



        # Get mask of cultivated area (N pixels)
        mask = cultivated_area[0, ...] > 0  # Shape: (H, W)
        N = numpy.count_nonzero(mask)  # Number of valid pixels

        # Create output array (B x N)
        soil_maps = numpy.zeros((len(existing_attributes), N), dtype=numpy.float32)

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

        return soil_maps  # Shape: (B, N)

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

                mask = resampled_cultivated_area > 0  # Shape: (H, W)
                masked_climate_stack = climate_stack[:, mask]  # Output shape: (B, N) where N = non-zero pixels
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
        root_dir = '/data2/hkaman/Data/FoundationModel'

        # Paths for each year
        self.cdl_paths = [os.path.join(root_dir, county_name, f"CDL/{yr}/{county_name}_{yr}.tif") for yr in self.year]
        self.landsat_dirs = [os.path.join(root_dir, county_name, f"Landsat/{yr}/") for yr in self.year]
        self.landsat_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.landsat_dirs)
        }
        self.et_dirs = [os.path.join(root_dir, county_name, f"ET/{yr}/") for yr in self.year]
        self.et_files = {
            yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
            for yr, dir in zip(self.year, self.et_dirs)
        }
        self.climate_paths = [os.path.join(root_dir, county_name, f"Climate/{yr}/DayMet_{county_name}_{yr}.nc") for yr in self.year]
        self.soil_dir = os.path.join(root_dir, county_name, "Soil/spatial/soilmu_a_ca053.shp")

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

    def get_reference_grid(self):
        """
        Create a reference grid based on the county geometry.

        Returns:
            tuple: (bounds, crs, resolution) of the reference grid.
        """
        county_geometry = self.get_county_geometry(self.county_name)
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
                crop_codes = [code for code, name in cdl_crop_legend.items() if name in crop_names]
                if not crop_codes:
                    raise ValueError("None of the specified crop names are valid.")
            else:
                crop_codes = [
                    code for code in cdl_crop_legend
                    if cdl_crop_legend[code] not in ["Non-agricultural", "NLCD-sampled categories", "Other"]
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
            climate_data = xr.open_dataset(climate_path)[variables]
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
        red_reprojected = numpy.empty((height, width), dtype=red_band.dtype)
        green_reprojected = numpy.empty((height, width), dtype=green_band.dtype)
        blue_reprojected = numpy.empty((height, width), dtype=blue_band.dtype)

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
    rgb = numpy.stack([red, green, blue], axis=-1)

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
    return xarray.open_dataset(file_path, engine="rasterio")#.to_array()

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
            array (numpy.ndarray): Inumpyut array to normalize.
        
        Returns:
            numpy.ndarray: Normalized array.
        """
        # Replace NaN values with the minimum of the non-NaN values
        nan_mask = numpy.isnan(array)
        if nan_mask.any():
            array[nan_mask] = numpy.nanmin(array)

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
    rgb = numpy.stack([red, green, blue], axis=-1)

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
    df = pandas.DataFrame(data, index=years)
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

