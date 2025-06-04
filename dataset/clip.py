import glob
import os 
import numpy
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import geopandas 
from rasterio.mask import mask
from rasterio.io import MemoryFile
import re
import geopandas
import dataset_wrapper as util

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


# a2_county_names = ['Inyo', 'Mono', 'San Benito', 'Santa Cruz', 'Santa Clara', 'San Mateo', 'Mariposa', 'Tuolumne']
a1_county_names = ['Riverside', 'San Diego', 'Ventura', 'Kern', 'Orange', 'San Bernardino', 'San Luis Obispo']


dataframe = geopandas.read_file(util.CA_COUNTIES_SHAPEFILE_DIR)

for p in range(67, 151):
    mosaic, out_meta, formatted_date = create_mosaic(f'/data2/hkaman/Data/FoundationModel/A/P{p}')
    year = formatted_date.split("_")[0] 

    for county in a1_county_names:
        
        normalized_county_name = county.strip().title() + " County"
        county_row = dataframe[dataframe["NAME"] == normalized_county_name]

        if county_row.empty:
            print(f"County '{county}' not found in shapefile.")
            continue

        geometry = county_row.geometry.values[0]

        # Use in-memory dataset to clip the mosaic
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as mosaic_dataset:
                mosaic_dataset.write(mosaic)
                clipped, clipped_transform = mask(dataset=mosaic_dataset, shapes=[geometry], crop=True)

                # Update metadata
                clipped_meta = out_meta.copy()
                clipped_meta.update({
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform
                })

                # Save the clipped file
                county_clean_name = county.replace(" ", "_")
                output_root = f'/data2/hkaman/Data/FoundationModel/Inputs/{county_clean_name}/Raw/Landsat/{year}'
                os.makedirs(output_root, exist_ok=True)
                output_filename = f"{county_clean_name}_{formatted_date}.tif"
                output_path = os.path.join(output_root, output_filename)

                # Only write the file if it does not already exist
                if not os.path.exists(output_path):
                    with rasterio.open(output_path, "w", **clipped_meta) as dest:
                        dest.write(clipped)
                    print(f"File {output_path} saved!")
                else:
                    print(f"File already exists, skipping: {output_path}")


                