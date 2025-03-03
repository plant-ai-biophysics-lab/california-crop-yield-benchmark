
import dataset_wrapper as util

import ee
ee.Authenticate() 
ee.Initialize()

import geopandas as gpd
dataframe = gpd.read_file('/data2/hkaman/Data/CDL/California_Counties.geojson')
dataframe = dataframe.to_crs(epsg=4326)

county_names = dataframe['NAME'].str[:-7].tolist()
county_names = county_names[34:]

for name in county_names: 
    for year in range(2008, 2023):
        if year != 2012: 
            dataset = util.DownloadClimateEE(year = year, county_name= name)()

