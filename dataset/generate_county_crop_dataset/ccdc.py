import os
import numpy as np
import geopandas as gpd
from typing import List
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import xarray as xr
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.transform import from_origin
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping
import geojson
from typing import List, Union
from skimage.transform import resize
import pandas as pd
from rapidfuzz import process, fuzz  # Faster alternative to fuzzywuzzy




cdl_crop_legend = {
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


def safe_read_csv(file_path, key_names, prefix):
    """Reads a CSV file safely, assigns unique prefixes to avoid merge conflicts, and extracts key columns."""
    try:
        df = pd.read_csv(file_path, delimiter="|", header=None, low_memory=False)

        # # Assign generic column names
        # df.columns = [f"{prefix}_Column_{i}" for i in range(df.shape[1])]

        # # Rename last two columns as keys (MUKEY, COKEY, CHKEY)
        # key_mapping = {df.columns[-(i+1)]: key for i, key in enumerate(reversed(key_names))}
        # df.rename(columns=key_mapping, inplace=True)

        return df
    

    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: {file_path} is missing or empty.")
        return pd.DataFrame()


class ProcessingYieldObs():
    def __init__(self, county_name: str):

        self.county_name = county_name
        self.folder_path = "/data2/hkaman/Data/FoundationModel/YieldObservation"
 
        self.manual_matches = {
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
        self.manual_matches["parsley "] = "Herbs" 

        self.crop_names = list(cdl_crop_legend.values())

        
    def __call__(self):

        root_folder = "/data2/hkaman/Data/FoundationModel"  
        all_dataframes = [] 
        
        for file_name in sorted(os.listdir(self.folder_path)):
            if file_name.endswith(".csv"):
                
                year = file_name.split("_")[-1].split(".")[0]

                
                file_path = os.path.join(self.folder_path, file_name)
                df = pd.read_csv(file_path)
    
                df = self.rename_column(df)

                df = self.match_crop_names(df)

                
                df.columns = df.columns.str.strip()
                
                df["key_crop_name"] = df.apply(lambda row: self.fix_no_match(row["crop_name"], row["key_crop_name"]), axis=1)
                df.loc[df["crop_name"].str.strip().str.lower() == "parsley", "key_crop_name"] = "Herbs"
                
                if self.county_name:
                    filtered_df = df[df['county'].str.strip().eq(self.county_name)]
                    
                    output_folder = os.path.join(root_folder, self.county_name, 'InD', year)
                    os.makedirs(output_folder, exist_ok=True)  

                    output_file = os.path.join(output_folder, f"yield_{year}.csv")

                    filtered_df.to_csv(output_file, index=False)

                    print(f"Processed and saved: {output_file}")

                else:
                    
                    all_dataframes.append(df)

        
        if not self.county_name and all_dataframes:
            return pd.concat(all_dataframes, ignore_index=True)

    def fix_no_match(self, crop_name, key_crop_name):
        """
        If key_crop_name is "No Match", replace it with the corresponding manual match.
        Ensures formatting consistency between crop_name and manual_matches.
        """
        if key_crop_name != "No Match":
            return key_crop_name  

        crop_name_clean = self.clean_text(crop_name).strip()  

       
        manual_matches_cleaned = {self.clean_text(k): v for k, v in self.manual_matches.items()}

        
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
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
        self.dataframe = self.dataframe.to_crs(epsg=int(self.target_crs.split(':')[-1]))

        # Base directory
        root_dir = '/data2/hkaman/Data/FoundationModel'
        self.output_dir = os.path.join(root_dir,county_name, f"InD/{self.year[0]}/{county_name}_{self.year[0]}.npz")

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
            
        np.savez_compressed(self.output_dir , input = outputs)

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

        aligned_data = np.zeros((source_data.shape[0], height, width), dtype=source_data.dtype)

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
            crop_code = next((code for code, name in cdl_crop_legend.items() if name == crop_name), None)

            # if crop_code is None:
            #     raise ValueError(f"Invalid crop name: {crop_name}. Please check the crop legend.")
            if crop_code is not None:
                # Create a mask for the selected crop

                mask = cdl_data == crop_code
                cultivated_area = np.where(mask, cdl_data, 0)

                # Align cultivated area to reference grid
                bounds, crs, resolution = self.get_reference_grid()
                aligned_cultivated_area, _, _ = self.align_to_geometry(
                    reference_bounds=bounds,
                    reference_crs=crs,
                    resolution=resolution,
                    source_data=np.expand_dims(cultivated_area, axis=0),
                    source_transform=cdl_transform,
                    source_crs=cdl_crs
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
        return np.stack(timeseries, axis=0) 

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


        return np.stack(timeseries, axis=0) 
    
    def get_soil_dataset(self, cultivated_area):
        """
        Rasterize SSURGO shapefile to match the reference grid and mask it with cultivated area.
        Extracts 10 key soil attributes from tabular data and returns a unique raster for each attribute.

        Returns:
            numpy.ndarray: A raster dataset of shape (B, N), where B is the number of soil attributes,
                        and N is the number of valid pixels (masked by cultivated_area).
        """

        # Load SSURGO spatial map unit polygons
        gdf = gpd.read_file(self.soil_dir)

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
        muaggatt_df = pd.read_csv(f"{tabular_dir}/muaggatt.csv")

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
            muaggatt_df["drclassdcd"] = muaggatt_df["drclassdcd"].map(drainage_mapping).astype(np.float32)

        if "hydgrpdcd" in muaggatt_df.columns:
            muaggatt_df["hydgrpdcd"] = muaggatt_df["hydgrpdcd"].map(soil_groups__mapping).astype(np.float32)



        # Get mask of cultivated area (N pixels)
        mask = cultivated_area[0, ...] > 0  # Shape: (H, W)
        N = np.count_nonzero(mask)  # Number of valid pixels

        # Create output array (B x N)
        soil_maps = np.zeros((len(existing_attributes), N), dtype=np.float32)

        for i, attr in enumerate(existing_attributes):
            attr_map = np.zeros_like(raster, dtype=np.float32)  

            for _, row in muaggatt_df.iterrows():
                mu_key = row["mukey"]
                value = row.get(attr, 0.0)  
                true_indices = np.where(raster == np.float32(mu_key))
                attr_map[true_indices] = value 
            # Extract only non-zero pixels where cultivated area > 0
            soil_maps[i, :] = attr_map[mask]
            soil_maps[i, np.isnan(soil_maps[i, :])] = 0.0

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

                climate_stack = np.stack(
                    [time_slice[var].values for var in variables],
                    axis=0
                )

                mask = resampled_cultivated_area > 0  # Shape: (H, W)
                masked_climate_stack = climate_stack[:, mask]  # Output shape: (B, N) where N = non-zero pixels
                timeseries.append(masked_climate_stack)

        return np.stack(timeseries, axis=0) 

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

        # Ensure the input is binary
        if not np.array_equal(cultivated_area, cultivated_area.astype(bool)):
            # raise ValueError("Input cultivated_area must already be binary.")
            cultivated_area = (cultivated_area > 0).astype(np.uint8)

        # Resample using maximum aggregation for binary values
        resampled_area = resize(
            cultivated_area.astype(np.float32),  # Use float32 for interpolation
            output_shape=target_shape,
            order=1,  # Bilinear interpolation to preserve the contribution of higher-resolution pixels
            anti_aliasing=False,
            preserve_range=True,
        )

        # Threshold: Any value > 0 means at least one contributing pixel was 1
        return (resampled_area > 0).astype(np.uint8)


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
        self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
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

        aligned_data = np.zeros((source_data.shape[0], height, width), dtype=source_data.dtype)

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

            mask = np.isin(cdl_data, crop_codes)
            cultivated_area = np.where(mask, cdl_data, 0)

            bounds, crs, resolution = self.get_reference_grid()
            aligned_cultivated_area, _, _ = self.align_to_geometry(
                reference_bounds=bounds,
                reference_crs=crs,
                resolution=resolution,
                source_data=np.expand_dims(cultivated_area, axis=0),
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
                mask = np.expand_dims(mask, axis=0)
                masked_landsat = aligned_landsat * mask
                timeseries.append(masked_landsat)

        return np.stack(timeseries, axis=0)

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
                mask = np.expand_dims(mask, axis=0)
                masked_et = aligned_et_data * mask
                timeseries.append(masked_et)

        return np.stack(timeseries, axis=0)

    def get_soil_dataset(self, cultivated_area):
        """
        Rasterize SSURGO shapefile to match the reference grid and mask it with cultivated area.
        """
        gdf = gpd.read_file(self.soil_dir)
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

        masked_raster = np.where(cultivated_area[0] > 0, raster, 0)
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

                climate_stack = np.stack(
                    [time_slice[var].values for var in variables],
                    axis=0
                )

                mask = resampled_cultivated_area > 0
                mask = np.expand_dims(mask, axis=0)
                masked_climate_stack = climate_stack * mask

                timeseries.append(masked_climate_stack)

        return np.stack(timeseries, axis=0)

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

        # Ensure the input is binary
        if not np.array_equal(cultivated_area, cultivated_area.astype(bool)):
            # raise ValueError("Input cultivated_area must already be binary.")
            cultivated_area = (cultivated_area > 0).astype(np.uint8)

        # Resample using maximum aggregation for binary values
        resampled_area = resize(
            cultivated_area.astype(np.float32),  # Use float32 for interpolation
            output_shape=target_shape,
            order=1,  # Bilinear interpolation to preserve the contribution of higher-resolution pixels
            anti_aliasing=False,
            preserve_range=True,
        )

        # Threshold: Any value > 0 means at least one contributing pixel was 1
        return (resampled_area > 0).astype(np.uint8)


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
        array = np.nan_to_num(array, nan=0)  # Replace NaN with 0
        # p5, p95 = np.percentile(array, [5, 95])  # Compute 5th and 95th percentiles
        # print(p5, p95)
        min = np.nanmin(array)
        max = np.nanmax(array)
        # array = np.clip(array, p5, p95)  # Clip values to the percentile range
        normalized = (array - min) / (max - min + 1e-10)  # Normalize to [0, 1]
        return np.clip(normalized * brightness, 0, 1) 

    red = normalize(landsat_data[2], brightness=3)  
    green = normalize(landsat_data[1], brightness=3)  
    blue = normalize(landsat_data[0], brightness=3)  

    # Stack RGB bands into an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)



    # # Identify fully white background (NaN or 0 in all datasets)
    # full_white_background = np.all(np.isnan(landsat_data) | (landsat_data == 0), axis=0)

    # # Identify light gray background (valid values outside main region)
    # valid_mask = ~full_white_background
    # light_gray_background = (landsat_data.sum(axis=0) > 0) & (~valid_mask)

    # # Landsat visualization with light gray and white backgrounds
    # visualization_image = np.ones_like(rgb_image)  # Initialize with white background
    # visualization_image[valid_mask] = rgb_image[valid_mask]  # Apply Landsat data to valid areas
    # visualization_image[light_gray_background] = [0.95, 0.95, 0.95]  # Light gray for valid areas outside the main region

    # # Replace NaN or 0 in other datasets with white or light gray backgrounds
    # def apply_background(data):
    #     data_with_bg = np.copy(data)
    #     data_with_bg[full_white_background] = np.nan  # Fully white for full background
    #     data_with_bg[light_gray_background] = 0  # Light gray for secondary background
    #     return data_with_bg

    # et_data = apply_background(et_data)
    # climate_data_resampled = apply_background(climate_data_resampled)
    # soil_data = apply_background(soil_data)

    # # Create colormap for soil data
    # unique_soil_types = np.unique(soil_data[~np.isnan(soil_data)])  # Only non-NaN values
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
    unique_soil_types = np.unique(soil_data[soil_data > 0])
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



# class DataDownloader:
#     def __init__(self, county_name: str = 'Monterey', year: List[int] | int = 2008, crop_names: str | List[str] | None = None):
#         """
#         Args:
#             county_name (str): The name of the county (e.g., 'Monterey').
#             year (list[int] or int): The year or list of years for data (e.g., 2008 or [2008, 2009]).
#             crop_names (str, list[str], or None): Name(s) of the crop(s) to filter (e.g., 'Corn' or ['Corn', 'Soybeans']).
#         """
#         self.county_name = county_name
#         self.year = year if isinstance(year, list) else [year]  # Ensure year is a list
#         self.crop_names = crop_names
#         self.target_crs = "EPSG:32610"
#         # Load California counties
#         self.dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
#         self.dataframe = self.dataframe.to_crs(epsg=self.target_crs)

#         # Base directory
#         root_dir = '/data2/hkaman/Data/FoundationModel'

#         # Paths for each year
#         self.cdl_paths = [os.path.join(root_dir, county_name, f"CDL/{yr}/{county_name}_{yr}.tif") for yr in self.year]
#         self.landsat_dirs = [os.path.join(root_dir, county_name, f"Landsat/{yr}/") for yr in self.year]
#         self.landsat_files = {
#             yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
#             for yr, dir in zip(self.year, self.landsat_dirs)
#         }
#         self.et_dirs = [os.path.join(root_dir, county_name, f"ET/{yr}/") for yr in self.year]
#         self.et_files = {
#             yr: sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
#             for yr, dir in zip(self.year, self.et_dirs)
#         }
#         self.climate_paths = [os.path.join(root_dir, county_name, f"Climate/{yr}/DayMet_{county_name}_{yr}.nc") for yr in self.year]
#         self.soil_dir = os.path.join(root_dir, county_name, "Soil/spatial/soilmu_a_ca053.shp")

    

#     def __call__(self, output_type: str | List[str] = "all", daily_climate: bool = True):
#         """
#         Args:
#             output_type (str or list[str]): Specify the desired output(s).
#                 - "all" for all outputs (default)
#                 - Any combination of: "landsat_data", "et_data", "climate_data", "soil_data"
#             daily_climate (bool): Whether to use daily climate data.

#         Returns:
#             dict or the requested subset of outputs.
#         """
#         # Ensure output_type is a list for consistent processing
#         if output_type == "all":
#             requested_outputs = ["landsat_data", "et_data", "climate_data", "soil_data"]
#         elif isinstance(output_type, str):
#             requested_outputs = [output_type]
#         elif isinstance(output_type, list):
#             requested_outputs = output_type
#         else:
#             raise ValueError("Invalid output_type. Must be 'all', a string, or a list of strings.")

#         # Validate requested outputs
#         valid_outputs = {"landsat_data", "et_data", "climate_data", "soil_data"}
#         invalid_outputs = [key for key in requested_outputs if key not in valid_outputs]
#         if invalid_outputs:
#             raise ValueError(f"Invalid output(s) requested: {invalid_outputs}. Valid options are {valid_outputs}.")

#         # Initialize cultivated area only once if needed
#         cdl_cultivated_data = None
#         if any(key in requested_outputs for key in valid_outputs):
#             cdl_cultivated_data = self.get_cultivated_area(crop_names=self.crop_names)

#         # Compute requested outputs
#         outputs = {}
#         if "landsat_data" in requested_outputs:
#             outputs["landsat_data"] = self.get_masked_landsat_timeseries(cdl_cultivated_data)
#         if "et_data" in requested_outputs:
#             outputs["et_data"] = self.get_masked_et_timeseries(cdl_cultivated_data)
#         if "climate_data" in requested_outputs:
#             outputs["climate_data"] = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
#         if "soil_data" in requested_outputs:
#             outputs["soil_data"] = self.get_soil_dataset(cdl_cultivated_data)

#         # Return requested outputs
#         if len(requested_outputs) == 1:
#             return outputs[requested_outputs[0]]  # Return single output directly
#         return outputs
    

#     def get_county_geometry(self, county_name):
#         """
#         Processes a county name to find the corresponding county in the DataFrame,
#         retrieves its geometry, and prepares it for Earth Engine operations.

#         Args:
#             county_name (str): The name of the county (case insensitive, without "County").

#         Returns:
#             tuple: (geometry, ee_geometry, county_name_modified) or None if not found.
#         """
#         # Normalize the input county name to match the DataFrame format
#         normalized_county_name = county_name.strip().title() + " County"

#         # Find the index of the matching county
#         try:
#             county_index = self.dataframe[self.dataframe["NAME"] == normalized_county_name].index[0]
#         except IndexError:
#             print(f"County '{county_name}' not found in the DataFrame.")
#             return

#         # Retrieve the geometry of the county
#         polygon = self.dataframe.iloc[county_index].geometry
#         aoi_geojson = geojson.Feature(geometry=mapping(polygon))
#         geometry = aoi_geojson["geometry"]

 
#         return geometry
    
#     def align_to_landsat(self, reference_crs, reference_transform, reference_shape, source_data, source_transform, source_crs):
#         """
#         Align source data (single or multi-band) to match the CRS, resolution, and shape of the reference (Landsat).

#         Parameters:
#             reference_crs: CRS of the reference data.
#             reference_transform: Transform of the reference data.
#             reference_shape: Shape of the reference data (height, width) as (H, W).
#             source_data: Multi-band or single-band data to align (numpy array).
#             source_transform: Transform of the source data.
#             source_crs: CRS of the source data.

#         Returns:
#             numpy.ndarray: Aligned data with the same shape as the reference (bands, H, W).
#         """
#         # Check if source_data is multi-band
#         if len(source_data.shape) == 3:  # Multi-band data (C, H, W)
#             num_bands = source_data.shape[0]
#         elif len(source_data.shape) == 2:  # Single-band data (H, W)
#             num_bands = 1
#             source_data = source_data[np.newaxis, ...]  # Add a band dimension
#         else:
#             raise ValueError("Source data must be 2D (H, W) or 3D (C, H, W).")

#         # Prepare an array for aligned data
#         aligned_data = np.zeros((num_bands, reference_shape[0], reference_shape[1]), dtype=source_data.dtype)

#         # Reproject each band of the source data to match Landsat
#         for band_index in range(num_bands):
#             reproject(
#                 source=source_data[band_index],
#                 destination=aligned_data[band_index],
#                 src_transform=source_transform,
#                 src_crs=source_crs,
#                 dst_transform=reference_transform,
#                 dst_crs=reference_crs,
#                 resampling=Resampling.bilinear
#             )

#         return aligned_data

#     def get_cultivated_area(self, crop_names=None):
#         """
#         Extract cultivated area and align it to the Landsat CRS and resolution.
#         """
#         with rasterio.open(self.cdl_paths[0]) as cdl_src:
#             cdl_data = cdl_src.read(1)
#             cdl_transform = cdl_src.transform
#             cdl_crs = cdl_src.crs

#             # Select crop codes
#             if crop_names:
#                 if not isinstance(crop_names, list):
#                     crop_names = [crop_names]
#                 crop_codes = [code for code, name in cdl_crop_legend.items() if name in crop_names]
#                 if not crop_codes:
#                     raise ValueError("None of the specified crop names are valid.")
#             else:
#                 crop_codes = [
#                     code for code in cdl_crop_legend
#                     if cdl_crop_legend[code] not in ["Non-agricultural", "NLCD-sampled categories", "Other"]
#                 ]

#             # Mask and extract cultivated area
#             mask = np.isin(cdl_data, crop_codes)
#             cultivated_area = np.where(mask, cdl_data, 0)

#             # Align to Landsat CRS and resolution
#             landsat_reference = self.landsat_files[self.year[0]][0]
#             with rasterio.open(landsat_reference) as landsat_src:
#                 aligned_cultivated_arlandsat_data = self.align_to_landsat(
#                     reference_crs=landsat_src.crs,
#                     reference_transform=landsat_src.transform,
#                     reference_shape=(landsat_src.height, landsat_src.width),
#                     source_data=cultivated_area,
#                     source_transform=cdl_transform,
#                     source_crs=cdl_crs
#                 )


#         height, width = aligned_cultivated_arlandsat_data[0].shape
#         aligned_bounds = rasterio.transform.array_bounds(height, width, landsat_src.transform)
#         return aligned_cultivated_arlandsat_data

#     def get_masked_landsat_timeseries(self, cultivated_area):
#         """
#         Mask Landsat timeseries imagery using the cultivated area and stack into a 4D matrix.
#         """
#         # band_indices = {5: [0, 1, 2, 3, 4, 5], 7: [0, 1, 2, 3, 4, 5], 8: [1, 2, 3, 4, 5, 6]}
#         if not self.landsat_files:
#             raise FileNotFoundError(f"No Landsat files found.")
#         timeseries = []

#         for landsat_path in self.landsat_files[self.year[0]]:
#             with rasterio.open(landsat_path) as src:
#                 landsat_data = src.read()
#                 landsat_transform = src.transform
#                 landsat_crs = src.crs
#                 # mission = 5
#                 # # mission = 5 if "LT05" in landsat_path else 7 if "LE07" in landsat_path else 8
#                 # if landsat_data.shape[0] > max(band_indices[mission]):
#                 #     selected_band_indices = band_indices[mission]
#                 #     landsat_data = landsat_data[selected_band_indices]
#                 # selected_band_indices = band_indices[mission]
#                 # landsat_data = landsat_data[selected_band_indices]

#                 # # Align cultivated area to Landsat CRS and resolution
#                 reprojected_cultivated_area = self.align_to_landsat(
#                     reference_crs=landsat_crs,
#                     reference_transform=landsat_transform,
#                     reference_shape=(src.height, src.width),
#                     source_data=cultivated_area,
#                     source_transform=src.transform,
#                     source_crs=src.crs
#                 )

#                 # Apply mask
#                 mask = reprojected_cultivated_area > 0
#                 mask = np.expand_dims(mask, axis=0)
#                 masked_landsat = landsat_data * mask
#                 timeseries.append(masked_landsat)
                
#         return np.stack(timeseries, axis=0)

#     def get_masked_et_timeseries(self, cultivated_area):
#         """
#         Mask ET timeseries imagery using the cultivated area and stack into a 4D matrix.
#         """
#         timeseries = []
#         landsat_reference = self.landsat_files[self.year[0]][0]

#         # Use Landsat as reference for alignment
#         with rasterio.open(landsat_reference) as landsat_src:
#             landsat_crs = landsat_src.crs
#             landsat_transform = landsat_src.transform
#             landsat_shape = (landsat_src.height, landsat_src.width)

#             # Align cultivated area to Landsat CRS and resolution once
#             aligned_cultivated_area = self.align_to_landsat(
#                 reference_crs=landsat_crs,
#                 reference_transform=landsat_transform,
#                 reference_shape=landsat_shape,
#                 source_data=cultivated_area,
#                 source_transform=landsat_transform,
#                 source_crs=landsat_crs
#             )

#             for et_path in self.et_files[self.year[0]]:
#                 with rasterio.open(et_path) as src:
#                     et_data = src.read()
#                     et_transform = src.transform
#                     et_crs = src.crs

#                     # Align ET data to Landsat CRS and resolution
#                     aligned_et_data = self.align_to_landsat(
#                         reference_crs=landsat_crs,
#                         reference_transform=landsat_transform,
#                         reference_shape=landsat_shape,
#                         source_data=et_data,
#                         source_transform=et_transform,
#                         source_crs=et_crs
#                     )

#                     # Apply mask
#                     mask = aligned_cultivated_area > 0
#                     mask = np.expand_dims(mask, axis=0)
#                     masked_et = aligned_et_data * mask
#                     timeseries.append(masked_et)

#         return np.stack(timeseries, axis=0)
    
#     def get_soil_dataset(self, cultivated_area):
#         """
#         Rasterize SSURGO shapefile to match Landsat spatial resolution (30m) and mask it with cultivated area.

#         Args:
#             cultivated_area (numpy.ndarray): Cultivated area mask aligned to Landsat resolution.

#         Returns:
#             numpy.ndarray: Masked rasterized soil data as a 2D numpy array.
#         """
#         gdf = gpd.read_file(self.soil_dir)

#         # Ensure the CRS matches Landsat
#         landsat_reference = self.landsat_files[self.year[0]][0]
#         with rasterio.open(landsat_reference) as ref:
#             transform = ref.transform
#             width = ref.width
#             height = ref.height
#             landsat_crs = ref.crs

#         gdf = gdf.to_crs(landsat_crs)  # Reproject shapefile to match Landsat CRS

#         # Map MUSYM values to unique integer codes
#         gdf["MUSYM_CODE"] = gdf["MUSYM"].astype("category").cat.codes

#         # Prepare shapes for rasterization
#         shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf["MUSYM_CODE"])]

#         # Rasterize the SSURGO data
#         raster = rasterize(
#             shapes=shapes,
#             out_shape=(height, width),
#             transform=transform,
#             dtype="int32"
#         )

#         # Mask the rasterized soil data with the cultivated area
#         masked_raster = np.where(cultivated_area > 0, raster, 0)  # Mask outside cultivated area with 0

#         return masked_raster
    
#     def resample_cultivated_area(self, cultivated_area, target_shape):
#         """
#         Resample the cultivated_area to match the target shape.

#         Args:
#             cultivated_area (numpy.ndarray): Binary mask for cultivated area (H, W).
#             target_shape (tuple): Desired shape (H, W) to match the climate data.

#         Returns:
#             numpy.ndarray: Resampled binary mask with the same shape as the climate data.
#         """
#         # Ensure the mask is binary (0 or 1)
#         from skimage.transform import resize
#         cultivated_area_binary = (cultivated_area > 0).astype(np.uint8)
        
#         # Resample using skimage's resize
#         resampled_area = resize(
#             cultivated_area_binary,
#             output_shape=target_shape,
#             order=0,  # Nearest-neighbor for binary data
#             anti_aliasing=False,
#             preserve_range=True,
#         )
        
#         # Threshold back to binary
#         resampled_area_binary = (resampled_area > 0).astype(np.uint8)
        
#         return resampled_area_binary

#     def get_climate_stack(self, cultivated_area, daily: bool = True):
#         """
#         Processes climate data from NetCDF files, resamples to 1km resolution, and masks with cultivated area.

#         Args:
#             cultivated_area (numpy.ndarray): Mask for the cultivated area at 30m resolution aligned with DayMet CRS.
#             daily (bool): If True, return daily observations (364 times). If False, return monthly means (12 times).

#         Returns:
#             numpy.ndarray: Climate data stack in format (T, C, H, W), where T is time, C is climate variables, and H, W are spatial dimensions.
#         """
#         variables = ['tmin', 'tmax', 'prcp', 'dayl', 'srad', 'vp', 'snow', 'pet']
#         timeseries = []

#         for climate_path in self.climate_paths:
#             # Load climate data with xarray
#             climate_data = xr.open_dataset(climate_path)[variables]
#             daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
#             climate_data = climate_data.rio.write_crs(daymet_crs)

#             # Reproject climate data to EPSG:32610
#             climate_data = climate_data.rio.reproject("EPSG:32610", resolution=(1000, 1000))
#             aligned_bounds = (591345.0, 3962475.0, 751815.0, 4086765.0)  # From previous calculation
#             climate_data = climate_data.rio.clip_box(*aligned_bounds)
#             # # Get the bounds and shape of the reprojected climate data
#             climate_bounds = climate_data.rio.bounds()

#             upscale_transform = from_bounds(*climate_bounds, width=climate_data.rio.width, height=climate_data.rio.height)
#             upscale_shape = (climate_data.rio.height, climate_data.rio.width)

#             target_shape = (climate_data.rio.height, climate_data.rio.width)

#             #    Resample cultivated area to match the climate data shape
#             # print(f"Resampling cultivated area from shape {cultivated_area.shape} to {target_shape}")
#             resampled_cultivated_area = self.resample_cultivated_area(cultivated_area[0, ...], target_shape)


#             # Aggregate to daily or monthly
#             if not daily:
#                 climate_data = climate_data.groupby("time.month").mean("time")
#                 time_steps = climate_data.month
#             else:
#                 time_steps = climate_data.time

#             # Process each time step
#             for time_step in time_steps:
#                 if not daily:
#                     time_slice = climate_data.sel(month=time_step)
#                 else:
#                     time_slice = climate_data.sel(time=time_step)

#                 # Stack variables into numpy array (C, H, W)
#                 climate_stack = np.stack(
#                     [time_slice[var].values for var in variables],
#                     axis=0
#                 )

#                 # Apply upscaled cultivated area mask
#                 mask = resampled_cultivated_area > 0
#                 mask = np.expand_dims(mask, axis=0)  # Add channel dimension
#                 masked_climate_stack = climate_stack * mask

#                 timeseries.append(masked_climate_stack)

#         return np.stack(timeseries, axis=0)

