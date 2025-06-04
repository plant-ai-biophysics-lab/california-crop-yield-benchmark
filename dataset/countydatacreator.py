import dataset_wrapper as util
import pandas as pd

County = ['SanBenito'] 

for c in County:
    for year in range(2022, 2023):

        if year != 2012: 
            m = pd.read_csv(f'/data2/hkaman/Data/YieldBenchmark/counties/{c}/InD/{year}/yield_{year}.csv')

            crop_names = m['key_crop_name'].unique().tolist()
            print(crop_names)

            output_vector = util.ModelProcessedDataModified(
                    county_name = c, 
                    year = year, 
                    crop_names= crop_names)(output_type = "all", 
                                                    daily_climate = True)
            
            print(f"{year} is done!")