import ccdc
import pandas as pd



for year in range(2008, 2023):
    if year != 2012: 
        m = pd.read_csv(f'/data2/hkaman/Data/FoundationModel/Monterey/InD/{year}/yield_{year}.csv')
        # Group by 'key_crop_name', summing 'yield' and collecting 'crop_name' into a list
        # m_grouped = m.groupby('key_crop_name').agg({
        #     'yield': 'sum',
        #     'crop_name': lambda x: list(x)  # Convert crop names to a list
        # }).reset_index()
        crop_names = m['key_crop_name'].unique().tolist()

        downloader = ccdc.ModelProcessedData(
            county_name = 'Monterey', 
            year = year, 
            crop_names = crop_names)

        output_dataset = downloader(output_type = "all", 
                                    daily_climate = True)
        

        print(f"{year} is done!")