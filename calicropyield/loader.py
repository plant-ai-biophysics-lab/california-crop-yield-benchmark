
import os
import json
import zipfile
import gdown
import numpy as np
import rasterio
from rasterio.mask import mask
from datetime import datetime


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


class DataDownloader:
    def __init__(self, target_dir: str):
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

        # Load Google Drive file IDs from bundled JSON
        links_path = os.path.join(os.path.dirname(__file__), "data_links.json")
        with open(links_path, "r") as f:
            self.data_links = json.load(f)

    def _download_single_tif(self, county, year, filename, file_id, dataset_type):
        url = f"https://drive.google.com/uc?id={file_id}"
        year_dir = os.path.join(self.target_dir, "counties", county, "data", dataset_type, str(year))
        os.makedirs(year_dir, exist_ok=True)
        output_path = os.path.join(year_dir, filename)

        if not os.path.exists(output_path):
            print(f"⬇️  Downloading {filename} for {county} {year}")
            gdown.download(url, output_path, quiet=False, fuzzy=True)
        else:
            print(f"✅ Already exists: {filename}")

        return output_path

    def download_CDL(self, county_name: list, year: list = None, crop_name: list = None, geometry=None):

        if year:
            year = [str(y) for y in year]
        else:
            year = [str(y) for y in range(2008, 2023) if y != 2012]

        CROP_NAME_TO_CODE = {v.lower(): k for k, v in CDL_CLASS_MAP.items()}
        crop_codes = [CROP_NAME_TO_CODE[c.lower()] for c in crop_name if c.lower() in CROP_NAME_TO_CODE] if crop_name else None

        log_path = os.path.join(self.target_dir, "logs", "cdl_download_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        for county in county_name:
            county_links = self.data_links.get("cdl", {}).get(county, {})
            for y in year:
                year_info = county_links.get(str(y))
                if not year_info:
                    print(f"⚠️ No data for {county} {y}")
                    continue

                filename = year_info["filename"]
                file_id = year_info["file_id"]
                src_path = self._download_single_tif(county, y, filename, file_id, dataset_type="cdl")
                self._process_raster(src_path, county, y, filename, crop_codes, geometry, log_path)

    def download_ET(self, county_name: list, year: list = None, geometry=None):
        self._generic_raster_download("et", county_name, year, geometry)

    def download_Landsat(self, county_name: list, year: list = None, geometry=None):
        self._generic_raster_download("landsat", county_name, year, geometry)

    def download_DayMet(self, county_name: list, year: list = None, variables: list = None, geometry=None):
        self._generic_raster_download("climate", county_name, year, geometry, variables)

    def download_soil(self, county_name: list, variable: list = None, spatial_resolution: str = None, geometry=None):
        for county in county_name:
            county_links = self.data_links.get("soil", {}).get(county, {})
            for fname, file_info in county_links.items():
                if variable and not any(v in fname for v in variable):
                    continue
                if spatial_resolution and spatial_resolution not in fname:
                    continue

                src_path = self._download_single_tif(county, "static", fname, file_info["file_id"], dataset_type="soil")
                self._process_raster(src_path, county, "static", fname, crop_codes=None, geometry=geometry, log_path=os.path.join(self.target_dir, "logs", "soil_download_log.jsonl"))

    def _generic_raster_download(self, dataset_type: str, county_name: list, year: list, geometry=None, variable_filter=None):
        log_path = os.path.join(self.target_dir, "logs", f"{dataset_type}_download_log.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if year:
            year = [str(y) for y in year]
        else:
            year = [str(y) for y in range(2008, 2023) if y != 2012]

        for county in county_name:
            county_links = self.data_links.get(dataset_type, {}).get(county, {})
            for y in year:
                year_info = county_links.get(str(y))
                if not year_info:
                    print(f"⚠️ No data for {county} {y}")
                    continue

                filename = year_info["filename"]
                if variable_filter and not any(v in filename for v in variable_filter):
                    continue

                file_id = year_info["file_id"]
                src_path = self._download_single_tif(county, y, filename, file_id, dataset_type)
                self._process_raster(src_path, county, y, filename, crop_codes=None, geometry=geometry, log_path=log_path)

    def _process_raster(self, src_path, county, year, filename, crop_codes, geometry, log_path):
        relpath = os.path.relpath(src_path, self.target_dir)
        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "county": county,
            "year": year,
            "filename": filename,
            "relpath": relpath,
            "geometry_used": geometry is not None,
        }

        try:
            with rasterio.open(src_path) as src:
                if geometry:
                    out_image, out_transform = mask(src, [geometry], crop=True)
                else:
                    out_image = src.read()
                    out_transform = src.transform

                if crop_codes:
                    mask_array = np.isin(out_image, crop_codes)
                    out_image = out_image * mask_array

                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                filtered_fname = filename.replace(".tif", "_filtered.tif")
                filtered_path = os.path.join(os.path.dirname(src_path), filtered_fname)
                with rasterio.open(filtered_path, "w", **out_meta) as dst:
                    dst.write(out_image)

                log_entry["status"] = "cropped" if geometry or crop_codes else "copied"
                log_entry["filtered_path"] = os.path.relpath(filtered_path, self.target_dir)

            os.remove(src_path)
            print(f"🗑️ Removed original: {src_path}")
        except Exception as e:
            log_entry["status"] = "error"
            log_entry["error"] = str(e)

        with open(log_path, "a") as logf:
            logf.write(json.dumps(log_entry) + "\n")

        print(f"  [{log_entry['status'].upper()}] {county} {year} → {filename}")