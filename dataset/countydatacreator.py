import dataset_wrapper as util
import pandas as pd



for year in range(2008, 2023):
    if year != 2012: 
        m = pd.read_csv(f'/data2/hkaman/Data/FoundationModel/Inputs/Napa/InD/{year}/yield_{year}.csv')

        crop_names = m['key_crop_name'].unique().tolist()
        print(crop_names)

        output_vector = util.ModelProcessedDataModified(
                county_name = 'Napa', 
                year = year, 
                crop_names= crop_names)(output_type = "all", 
                                                daily_climate = True)
        
        print(f"{year} is done!")