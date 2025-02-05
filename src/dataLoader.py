import os
import numpy as np
import pandas as pd
# from src.configs import set_seed
# set_seed(0)

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

from torch.utils.data import Dataset


EXTREME_LOWER_THRESHOLD = 9  #22.24
EXTREME_UPPER_THRESHOLD = 22 #54.36
HECTARE_TO_ACRE_SCALE = 2.471 # 2.2417

def _preprocess_df(df):
    """pre=processing includes three steps: 
        1. removing the crops with harvest area equal to 0
        2. removing the crops with yield 0 
        4. remove county called "State Total"
    
    """

    df = df[df['harvest_acres'] != 0]
    df = df[df['yield'] > 0]
    df = df[df['county'] != "State Total"]

    return df



class TT:

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
            try:
                # Get cultivated area for the current crop
                cdl_cultivated_data = self.get_cultivated_area(crop_name=crop_name)

                # Initialize the crop-specific dictionary
                crop_outputs = {}

                # Compute requested outputs for this crop
                if "landsat_data" in requested_outputs:
                    crop_outputs["landsat_data"] = self.get_masked_landsat_timeseries(cdl_cultivated_data)
                if "et_data" in requested_outputs:
                    crop_outputs["et_data"] = self.get_masked_et_timeseries(cdl_cultivated_data)
                if "climate_data" in requested_outputs:
                    crop_outputs["climate_data"] = self.get_climate_stack(cdl_cultivated_data, daily=daily_climate)
                if "soil_data" in requested_outputs:
                    crop_outputs["soil_data"] = self.get_soil_dataset(cdl_cultivated_data)

                # Store the crop's output in the main dictionary
                outputs[crop_name] = crop_outputs
            
            except ValueError as e:
                print(f"Warning: {e} - Skipping crop '{crop_name}'")

        # Return single output directly if only one crop was requested
        if len(self.crop_names) == 1:
            return outputs[self.crop_names[0]]

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
            if crop_code is None:
                raise ValueError(f"Invalid crop name: {crop_name}. Please check the crop legend.")

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




def read_and_split_csf_files(base_path):

    train_years = list(range(2008, 2012)) + list(range(2013, 2019))  # Excludes 2012
    valid_years = [2019, 2020]
    test_years = [2021, 2022]

    train_df, valid_df, test_df = [], [], []

    for year in range(2008, 2023):  # Loop through all expected years
        folder_path = os.path.join(base_path, str(year))
        csv_file = os.path.join(folder_path, f"yield_{year}.csv")  # Assuming filename matches the year

        if year == 2012 or not os.path.exists(csv_file):
            continue  # Skip missing years or non-existing files

        df = pd.read_csv(csv_file)  # Update if a different format is needed
        df = df[df['key_crop_name'] != 'No Match']

        if year in train_years:
            train_df.append(df)
        elif year in valid_years:
            valid_df.append(df)
        elif year in test_years:
            test_df.append(df)

    # Concatenate dataframes
    train_df = pd.concat(train_df, ignore_index=True) if train_df else None
    valid_df = pd.concat(valid_df, ignore_index=True) if valid_df else None
    test_df = pd.concat(test_df, ignore_index=True) if test_df else None

    return train_df, valid_df, test_df
    


def dataloader(county_name: str = 'Monterey'):


    base_csv_path = f'/data2/hkaman/Data/FoundationModel/{county_name}/InD/'

    train_df, valid_df, test_df = read_and_split_csf_files(base_csv_path)
    print(train_df.shape, valid_df.shape, test_df.shape)

    DataGen(df = train_df).__getitem__(20)






class DataGen(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.base_path = '/data2/hkaman/Data/FoundationModel'
        
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        year = self.df.loc[idx, 'year'] 
        crop_name = self.df.loc[idx, 'key_crop_name'].strip()
        county = self.df.loc[idx, 'county'].strip()

        npz_file_path = os.path.join(self.base_path, f'{county}', 'InD', f'{year}', f'{county}_{year}.npz')
        loaded_data = np.load(npz_file_path, allow_pickle=True)["input"]
        loaded_data = loaded_data.item()  # Convert the array to a dictionary
        

        landsat = loaded_data[crop_name]['landsat_data']
        et = loaded_data[crop_name]['et_data']
        climate = loaded_data[crop_name]['climate_data']
        soil = loaded_data[crop_name]['soil_data']

        # Original label
        original_y = self.df.loc[idx]['yield']


        # print(landsat.shape, et.shape, climate.shape, soil.shape, original_y)

        return 
    
    def normalize(self):
        
    







def get_dataloaders(
        batch_size:int, 
        exp_name: str): 

    
    root_exp_dir = '/data2/hkaman/Projects/Foundational/' 
    exp_output_dir = root_exp_dir + 'EXPs/' + 'EXP_' + exp_name

    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(os.path.join(exp_output_dir, 'loss'))

    train_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/train.csv', index_col=0)
    valid_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/val.csv', index_col= 0)
    test_csv  = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/test.csv', index_col= 0)

    print(f"{train_csv.shape} | {valid_csv.shape} | {test_csv.shape}")
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#
    #csv_coord_dir = '/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/'


    dataset_training = Sentinel_Dataset(
        train_csv
    )

    dataset_validate = Sentinel_Dataset(
        valid_csv   
    )
    
    dataset_test = Sentinel_Dataset(
        test_csv
    )     

    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
  
    data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                    shuffle=True,  num_workers=8) 
    data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                    shuffle=False, num_workers=8)  
    data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                    shuffle=False, num_workers=8) 

    return data_loader_training, data_loader_validate, data_loader_test


class Sentinel_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df.reset_index(inplace=True, drop=True)
        self.wsize = 16
        
        # Create a mapping for unique labels
        unique_labels = sorted(self.df['cultivar_id'].unique())
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.df)
    
    def _crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        if src.ndim == 2:
            src = np.expand_dims(src, axis=0)
            src = np.expand_dims(src, axis=-1)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
    
    def histogram_equalization_4d(self, image):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        eq_image = np.empty_like(image)
        for c in range(image.shape[0]):
            for t in range(image.shape[3]):
                eq_image[c, :, :, t] = cv2.equalizeHist(image[c, :, :, t])
        return eq_image

    def __getitem__(self, idx):
        xcoord = self.df.loc[idx]['X'] 
        ycoord = self.df.loc[idx]['Y'] 
        S2_path = self.df.loc[idx]['IMG_PATH']
        x = self._crop_gen(S2_path, xcoord, ycoord) 
        x = np.swapaxes(x, -1, 0)   
        x = self.histogram_equalization_4d(x)
        x = x / 255.0
        x = torch.as_tensor(x, dtype=torch.float32)

        # Original label
        original_y = self.df.loc[idx]['cultivar_id']
        # Remap label to a consecutive range
        y = self.label_mapping[original_y]
        y = torch.tensor(y, dtype=torch.long)

        return x, y
