from pathlib import Path
import os
import json
import zipfile
import gdown
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from datetime import datetime
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import service_account





CDL_CLASS_MAP = {
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


DRIVER_ROOT_ID = "1Ci_LlLF1-hcLt898CTbyYnYTFPZNWlUe"
CREDENTIAL_PATH ="tokens/service_account.json"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class DataDownloader:
    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        self.drive_root_id = DRIVER_ROOT_ID
        os.makedirs(self.target_dir, exist_ok=True)
        self.service = self.authenticate_drive()

    def authenticate_drive(self):
        creds_path = Path(__file__).parent / CREDENTIAL_PATH

        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return build('drive', 'v3', credentials=credentials)

    def get_subfolder_id(self, parent_id, folder_name):
        response = self.service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_id}' in parents and trashed=false",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        folders = response.get("files", [])
        return folders[0]["id"] if folders else None

    def find_file(self, name, parent_folder_id):
        results = self.service.files().list(
            q=f"name='{name}' and '{parent_folder_id}' in parents and trashed=false",
            fields="files(id, name)").execute()
        items = results.get('files', [])
        return items[0]['id'] if items else None

    def _download_single_file(self, county, year, filename, dataset_type):
        county_id = self.get_subfolder_id(self.drive_root_id, county)
        if not county_id: return None
        data_id = self.get_subfolder_id(county_id, "data")
        if not data_id: return None
        dataset_id = self.get_subfolder_id(data_id, dataset_type)
        if not dataset_id: return None

        if dataset_type == "soil":
            file_id = self.find_file(filename, dataset_id)
            if not file_id:
                print(f"❌ File not found: {filename}")
                return None
            
            sub_dir = os.path.join(self.target_dir, "counties", county, "data", dataset_type)
        else:
            year_id = self.get_subfolder_id(dataset_id, year)
            if not year_id: return None

            file_id = self.find_file(filename, year_id)
            if not file_id:
                print(f"❌ File not found: {filename}")
                return None
            
            sub_dir = os.path.join(self.target_dir, "counties", county, "data", dataset_type, str(year))
        

        os.makedirs(sub_dir, exist_ok=True)
        output_path = os.path.join(sub_dir, filename)

        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"⬇️ Downloading {filename}")
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"✅ Already exists: {output_path}")

        return output_path

    def download_ET(self, county_names: list, years: list = None, geometry=None):
        if years is None:
            years = [str(y) for y in range(2008, 2023) if y != 2012]
        else:
            years = [str(y) for y in years]

        for county in county_names:
            for year in years:
                for month in range(1, 13):
                    month_str = f"{month:02d}"  # ensures leading zero
                    filename = f"{county}_OpenET_{year}_{month_str}.tif"
                    path = self._download_single_file(county, year, filename, dataset_type="et")

                    if path and geometry:
                        cropped_path = path.replace(".tif", "_cropped.tif")
                        success = self._geometry_cropping_tif(geometry=geometry, src_path=path, save_path=cropped_path)
                        if success:
                            os.remove(path)
                            print(f"🗑️ Removed original: {path}")
                            print(f"✅ Saved cropped: {cropped_path}")
    
    def download_Landsat(self, county_names: list, years: list = None, geometry=None):
        if years is None:
            years = [str(y) for y in range(2008, 2023) if y != 2012]
        else:
            years = [str(y) for y in years]

        for county in county_names:
            for year in years:
                for month in range(1, 13):
                    month_str = f"{month:02d}"  # ensures leading zero
                    filename = f"{county}_LT_{year}_{month_str}.tif"
                    path = self._download_single_file(county, year, filename, dataset_type="landsat")

                    if path and geometry:
                        cropped_path = path.replace(".tif", "_cropped.tif")
                        success = self._geometry_cropping_tif(geometry=geometry, src_path=path, save_path=cropped_path)
                        if success:
                            os.remove(path)
                            print(f"🗑️ Removed original: {path}")
                            print(f"✅ Saved cropped: {cropped_path}")

    def _geometry_cropping_tif(self, geometry, src_path, save_path):
        try:
            with rasterio.open(src_path) as src:
                out_image, out_transform = mask(src, [geometry], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                with rasterio.open(save_path, "w", **out_meta) as dst:
                    dst.write(out_image)
            return True
        except Exception as e:
            print(f"❌ Cropping failed for {src_path}: {e}")
            return False

    def download_Climate(self, county_names: list, years: list = None, geometry=None, variables=None):
        if years is None:
            years = [str(y) for y in range(2008, 2023) if y != 2012]
        else:
            years = [str(y) for y in years]

        for county in county_names:
            for year in years:
                filename = f"{county}_DayMet_{year}.nc"
                path = self._download_single_file(county, year, filename, dataset_type="climate")

                if path and geometry:
                    cropped_path = path.replace(".nc", "_cropped.nc")
                    success = self._geometry_cropping_netcdf(geometry=geometry, src_path=path, save_path=cropped_path, variables = variables)
                    if success:
                        os.remove(path)
                        print(f"🗑️ Removed original: {path}")
                        print(f"✅ Saved cropped: {cropped_path}")

    def _geometry_cropping_netcdf(self, geometry, src_path, save_path, variables=None):
        try:
            # Load dataset
            ds = xr.open_dataset(src_path)

            # Select specific variables if requested
            if variables is not None:
                ds = ds[variables]

            # Convert geometry to GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")  # Adjust CRS if needed

            # Reproject GeoDataFrame to match NetCDF's CRS
            if 'crs' in ds.attrs:
                gdf = gdf.to_crs(ds.attrs['crs'])

            # Get bounds from geometry
            bounds = gdf.total_bounds
            minx, miny, maxx, maxy = bounds

            # Crop the dataset using .sel
            cropped = ds.sel(
                lon=slice(minx, maxx),
                lat=slice(miny, maxy)
            )

            # Save the cropped NetCDF
            cropped.to_netcdf(save_path)

            print(f"✅ Cropped and saved: {save_path}")
            return True

        except Exception as e:
            print(f"❌ Cropping NetCDF failed for {src_path}: {e}")
            return False

    def download_Soil(self, county_names: list, geometry=None, variables=None):

        for county in county_names:

            filename = f"{county}_soil.nc"
            path = self._download_single_file(county, year = None, filename = filename, dataset_type="soil")

            if path and geometry:
                cropped_path = path.replace(".nc", "_cropped.nc")
                success = self._geometry_cropping_netcdf(geometry=geometry, src_path=path, save_path=cropped_path, variables = variables)
                if success:
                    os.remove(path)
                    print(f"🗑️ Removed original: {path}")
                    print(f"✅ Saved cropped: {cropped_path}")

    def download_CDL(self, county_names: list, years: list = None, geometry=None, crop_types: list = None):
        if years is None:
            years = [str(y) for y in range(2008, 2023) if y != 2012]
        else:
            years = [str(y) for y in years]

        for county in county_names:
            for year in years:
                filename = f"{county}_CDL_{year}.TIF"
                path = self._download_single_file(county, year, filename, dataset_type="cdl")

                if path:
                    cropped_path = path.replace(".TIF", "_cropped.TIF")
                    success = self._process_cdl(src_path=path, save_path=cropped_path, geometry=geometry, crop_types=crop_types)
                    if success:
                        os.remove(path)
                        print(f"🗑️ Removed original: {path}")
                        print(f"✅ Saved processed CDL: {cropped_path}")

    def _process_cdl(self, src_path, save_path, geometry=None, crop_types=None):
        try:
            with rasterio.open(src_path) as src:
                cdl_data = src.read(1)
                cdl_meta = src.meta.copy()
                transform = src.transform

                # Geometry-based masking (optional)
                if geometry:
                    out_image, out_transform = mask(src, [geometry], crop=True)
                    cdl_data = out_image[0]
                    cdl_meta.update({
                        "height": cdl_data.shape[0],
                        "width": cdl_data.shape[1],
                        "transform": out_transform
                    })

                # Crop type filtering (optional)
                if crop_types:
                    crop_type_codes = [code for code, name in CDL_CLASS_MAP.items() if name in crop_types]
                    mask_array = np.isin(cdl_data, crop_type_codes)
                    cdl_data = np.where(mask_array, cdl_data, 0)

            # Write the processed raster
            with rasterio.open(save_path, "w", **cdl_meta) as dst:
                dst.write(cdl_data, 1)

            return True

        except Exception as e:
            print(f"❌ Failed to process CDL {src_path}: {e}")
            return False
