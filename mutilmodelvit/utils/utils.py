import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter




list_crop_names = ['ALMONDS ALL', 'ANISE (FENNEL)', 'APPLES ALL', 'APRICOTS ALL', 'ARTICHOKES',
 'ASPARAGUS UNSPECIFIED', 'AVOCADOS ALL', 'BARLEY FEED', 'BARLEY SEED',
 'BARLEY UNSPECIFIED', 'BEANS BLACKEYE (PEAS)',
 'BEANS DRY EDIBLE UNSPECIFIED', 'BEANS FAVA', 'BEANS FRESH UNSPECIFIED',
 'BEANS GARBANZO', 'BEANS KIDNEY RED', 'BEANS LIMA BABY DRY',
 'BEANS LIMA LARGE DRY', 'BEANS LIMA UNSPECIFIED', 'BEANS PINK', 'BEANS SEED',
 'BEANS SNAP FRESH MARKET', 'BEANS SNAP UNSPECIFIED', 'BEETS GARDEN',
 'BERRIES BLACKBERRIES', 'BERRIES BLUEBERRIES', 'BERRIES BOYSENBERRIES',
 'BERRIES BUSHBERRIES UNSPECIFIED', 'BERRIES RASPBERRIES',
 'BERRIES STRAWBERRIES FRESH MARKET', 'BERRIES STRAWBERRIES PROCESSING',
 'BERRIES STRAWBERRIES UNSPECIFIED', 'BROCCOLI FOOD SERVICE',
 'BROCCOLI FRESH MARKET', 'BROCCOLI PROCESSING', 'BROCCOLI UNSPECIFIED',
 'BRUSSELS SPROUTS', 'CABBAGE CHINESE & SPECIALTY', 'CABBAGE HEAD',
 'CARROTS FOOD SERVICE', 'CARROTS FRESH MARKET', 'CARROTS PROCESSING',
 'CARROTS UNSPECIFIED', 'CAULIFLOWER FOOD SERVICE',
 'CAULIFLOWER FRESH MARKET', 'CAULIFLOWER UNSPECIFIED',
 'CELERY FOOD SERVICE', 'CELERY FRESH MARKET' ,'CELERY UNSPECIFIED',
 'CHAYOTES' 'CHERIMOYAS', 'CHERRIES SWEET', 'CHESTNUTS', 'CHIVES', 'CILANTRO',
 'CITRUS UNSPECIFIED' 'COLLARD GREENS' 'CORN GRAIN' 'CORN POPCORN'
 'CORN SILAGE' 'CORN SWEET ALL' 'COTTON LINT PIMA'
 'COTTON LINT UNSPECIFIED' 'COTTON LINT UPLAND' 'COTTON SEED PLANTING'
 'COTTONSEED' 'CUCUMBERS' 'DATES' 'EGGPLANT ALL' 'ENDIVE ALL'
 'ESCAROLE ALL' 'FIELD CROP BY-PRODUCTS' 'FIELD CROPS SEED MISC.'
 'FIELD CROPS UNSPECIFIED' 'FIGS DRIED' 'FLOWERS FOLIAGE CUT ALL'
 'FRUITS & NUTS UNSPECIFIED' 'GARLIC ALL' 'GRAPEFRUIT ALL' 'GRAPES RAISIN'
 'GRAPES TABLE' 'GRAPES UNSPECIFIED' 'GRAPES WINE'
 'GREENS TURNIP & MUSTARD' 'GUAVAS' 'HAY ALFALFA' 'HAY GRAIN'
 'HAY GREEN CHOP' 'HAY OTHER UNSPECIFIED' 'HAY SUDAN' 'HAY WILD'
 'HORSERADISH' 'KALE' 'KIWIFRUIT' 'KOHLRABI' 'KUMQUATS' 'LEEKS'
 'LEMONS ALL' 'LETTUCE BULK SALAD PRODUCTS' 'LETTUCE HEAD' 'LETTUCE LEAF'
 'LETTUCE ROMAINE' 'LETTUCE UNSPECIFIED' 'LIMES ALL' 'MACADAMIA NUTS'
 'MELONS CANTALOUPE' 'MELONS CRENSHAW' 'MELONS HONEYDEW'
 'MELONS UNSPECIFIED' 'MELONS WATERMELON' 'MINT' 'MUSHROOMS' 'NECTARINES'
 'NURSERY PLANTS STRAWBERRY' 'NURSERY PRODUCTS MISC.'
 'NURSERY FLOWER SEEDS' 'NURSERY FRUIT/VINE/NUT NON-BEARING'
 'NURSERY TURF' 'NURSERY WOODY ORNAMNTALS' 'OATS GRAIN' 'OKRA' 'OLIVES'
 'ONIONS' 'ONIONS GREEN & SHALLOT' 'ORANGES NAVEL' 'ORANGES UNSPECIFIED'
 'ORANGES VALENCIA' 'PARSLEY' 'PARSNIPS' 'PASTURE FORAGE MISC.'
 'PASTURE IRRIGATED' 'PASTURE RANGE' 'PEACHES CLINGSTONE'
 'PEACHES FREESTONE' 'PEACHES UNSPECIFIED' 'PEANUTS ALL' 'PEARS ASIAN'
 'PEARS BARTLETT' 'PEARS PRICKLY' 'PEARS UNSPECIFIED' 'PEAS DRY EDIBLE'
 'PEAS EDIBLE POD (SNOW)' 'PEAS GREEN UNSPECIFIED' 'PECANS' 'PEPPERS BELL'
 'PEPPERS CHILI HOT' 'PERSIMMONS' 'PISTACHIOS' 'PLUMCOTS' 'PLUMS'
 'PLUMS DRIED' 'POMEGRANATES' 'POTATOES ALL' 'POTATOES SWEET' 'PUMPKINS'
 'QUINCE' 'RADICCHIO' 'RADISHES' 'RAPPINI' 'RICE MILLING' 'RICE SEED'
 'RICE WILD' 'RUTABAGAS' 'RYE GRAIN' 'RYE SEED' 'SAFFLOWER'
 'SAFFLOWER SEED PLANTING' 'SALAD GREENS MISC.' 'SEED ALFALFA'
 'SEED BERMUDA GRASS' 'SEED CLOVER UNSPECIFIED' 'SEED GRASS UNSPECIFIED'
 'SEED OTHER (NO FLOWERS)' 'SEED SUDAN GRASS' 'SEED VEGETABLE & VINECROP'
 'SEED VETCH' 'SILAGE' 'SORGHUM GRAIN' 'SORGHUM SILAGE' 'SPICES AND HERBS'
 'SPINACH FOOD SERVICE' 'SPINACH FRESH MARKET' 'SPINACH PROCESSING'
 'SPINACH UNSPECIFIED' 'SQUASH' 'STRAW' 'SUGAR BEETS'
 'SUNFLOWER SEED PLANTING' 'SWISS CHARD' 'TANGELOS'
 'TANGERINES & MANDARINS' 'TARO ROOT' 'TOMATILLO' 'TOMATOES FRESH MARKET'
 'TOMATOES PROCESSING' 'TOMATOES UNSPECIFIED' 'TRITICALE' 'TURNIPS ALL'
 'VEGETABLES ORIENTAL ALL' 'VEGETABLES UNSPECIFIED' 'WALNUTS BLACK'
 'WALNUTS ENGLISH' 'WHEAT ALL' 'WHEAT SEED' 'CHRISTMAS TREES & CUT GREENS'
 'FLOWERS ASTERS CUT' 'FLOWERS CACTI SUCCULENTS'
 'FLOWERS CARNATION CUT MINIATURE' 'FLOWERS CARNATION UNSPECIFIED'
 'FLOWERS CHRYSANTHEMUM UNSPECIFIED' 'FLOWERS CUT UNSPECIFIED'
 'FLOWERS DECORATIVE DRIED' 'FLOWERS FOLIAGE PLANTS' 'FLOWERS IRISES CUT'
 'FLOWERS LILACS CUT' 'FLOWERS POINSETTIA POTTED'
 'FLOWERS ROSES UNSPECIFIED' 'NURSERY PLANTS BEDDING'
 'NURSERY PLANTS ORCHID' 'NURSERY PLANTS POTTED UNSPECIFIED'
 'NURSERY PLANTS ROSE' 'NURSERY PLANTS VEGETABLE BEDDING'
 'NURSERY BULBS LILY' 'NURSERY FLOWER PROPAGATIVE MATERIALS'
 'NURSERY GERANIUMS' 'NURSERY HERBACIOUS PERERNNIALS'
 'NURSERY HORTICULTRAL SPECIMIN MISC.' 'SPROUTS ALFALFA & BEAN' 'JOJOBA'
 'OATS SEED' 'POTATOES SEED' 'RYEGRASS PERENNIAL ALL'
 'FLOWERS ORCHIDS CUT ALL' 'SOYBEANS' 'TOMATOES CHERRY' 'ALMOND HULLS'
 'YUCCA' 'CATTLE & CALVES UNSPECIFIED' 'CATTLE BEEF COW BREEDING'
 'CORN CRAZY' 'FLOWERS CHRYSANTHEMUM CUT POM.' 'GAME BIRDS UNSPEC'
 'GOATS & KIDS UNSPECIFIED' 'HOGS & PIGS UNSPECIFIED'
 'JERUSALEM ARTICHOKES' 'PHEASANTS' 'PIGEONS & SQUABS'
 'POULTRY UNSPECIFIED' 'RABBITS' 'SEEDS MISC. OIL UNSPECIFIED'
 'SHEEP & LAMBS UNSPECIFIED' 'BEANS LIMA GREEN' 'BEETS'
 'BERRIES STRAWBERRIES ALL' 'CAROBS' 'CUCUMBERS GREENHOUSE'
 'FIELD CROPS SEED MISC' 'FIGS' 'FLOWERS FLOWERING & FOLIAGE PLANTS'
 'HEMP UNSPECIFIED' 'HOPS' 'NURSERY PRODUCTS MISC'
 'NURSERY WOODY ORNAMENTALS' 'PASTURE FORAGE MISC'
 'PLUMCOTS & OTHER HYBRID STONE FRUIT' 'SALAD GREENS MISC'
 'VEGETABLES ASIAN' 'ASPARAGUS FRESH MARKET' 'BARLEY MALTING'
 'BEANS DRY EDIBLE UNSPEC.' 'BEANS LIMA LG. DRY'
 'BERRIES BUSHBERRIES UNSPEC.' 'BERRIES STRAWBERRIES F MKT'
 'BERRIES STRAWBERRIES PROC.' 'BERRIES STRAWBERRIES UNSPEC'
 'CABBAGE CH. & SPECIALTY' 'CORN WHITE' 'LETTUCE BULK SALAD PRODS.'
 'PEAS GREEN PROCESSING' 'PEAS SEED' 'POTATOES IRISH ALL'
 'SALAD GREENS NEC.' 'SEED VEG & VINECROP' 'APIARY PRODUCTS POLLIN. FEES'
 'BEANS SNAP PROCESSING' 'BERRIES LOGANBERRIES' 'CABBAGE RED'
 'CAULIFLOWER PROCESSING' 'MUSTARD' 'NURSERY FRT/VINE/NUT N-BEAR'
 'WATERCRESS' 'WOOL' 'CORN SEED' 'VEGETABLES BABY' 'TOMATOES GREENHOUSE'
 'FOOD GRAINS MISC.' 'NURSERY FLOWER BULBS/CORMS/RHIZOMES'
 'FLOWERS ROSES CUT STANDARD' 'GUAR' 'ASPARAGUS PROCESSING'
 'BEANS RED SMALL' 'BERRIES OLALLIEBERRIES' 'CELERY PROCESSING' 'FEIJOA'
 'MELONS CASABA' 'PEAS GREEN FRESH MARKET' 'RHUBARB' 'RICE SWEET'
 'SEED LADINO CLOVER' 'SUNFLOWER SEED' 'CARDOON' 'POMELO'
 'VEGETABLES GREENHOUSE' 'NURSERY FL. PROPG. MTRLS'
 'CITRUS BY-PRODUCTS MISC.' 'FOREST PRODUCTS NURSERY PROD'
 'BEANS WHITE SMALL' 'CORN POPCORN SEED' 'BEANS PINTO'
 'APIARY PRODUCTS BEESWAX' 'APIARY PRODUCTS HONEY'
 'FLOWERS CARNATION CUT MIN.' 'FLOWERS CARNATION CUT STD.'
 'FLOWERS ROSES CUT MIN.' 'FOREST PRODUCTS FIREWOOD'
 'NURSERY HERBAC. PRRNLS' 'FOREST PRODUCTS LUMBER'
 'APIARY PRODUCTS BEES UNSPEC' 'PIMENTOS' 'WILD-LIFE MANAGEMENT']

crop_dict = {
    "1": "Corn",
    "2": "Cotton",
    "3": "Rice",
    "4": "Sorghum",
    "5": "Soybeans",
    "6": "Sunflower",
    "10": "Peanuts",
    "11": "Tobacco",
    "12": "Sweet Corn",
    "13": "Pop or Orn Corn",
    "14": "Mint",
    "21": "Barley",
    "22": "Durum Wheat",
    "23": "Spring Wheat",
    "24": "Winter Wheat",
    "25": "Other Small Grains",
    "26": "Dbl Crop WinWht/Soybeans",
    "27": "Rye",
    "28": "Oats",
    "29": "Millet",
    "30": "Speltz",
    "31": "Canola",
    "32": "Flaxseed",
    "33": "Safflower",
    "34": "Rape Seed",
    "35": "Mustard",
    "36": "Alfalfa",
    "37": "Other Hay/Non Alfalfa",
    "38": "Camelina",
    "39": "Buckwheat",
    "41": "Sugarbeets",
    "42": "Dry Beans",
    "43": "Potatoes",
    "44": "Other Crops",
    "45": "Sugarcane",
    "46": "Sweet Potatoes",
    "47": "Misc Vegs & Fruits",
    "48": "Watermelons",
    "49": "Onions",
    "50": "Cucumbers",
    "51": "Chick Peas",
    "52": "Lentils",
    "53": "Peas",
    "54": "Tomatoes",
    "55": "Caneberries",
    "56": "Hops",
    "57": "Herbs",
    "58": "Clover/Wildflowers",
    "59": "Sod/Grass Seed",
    "60": "Switchgrass",
    "61": "Fallow/Idle Cropland",
    "62": "Pasture/Grass",
    "63": "Forest",
    "64": "Shrubland",
    "65": "Barren",
    "66": "Cherries",
    "67": "Peaches",
    "68": "Apples",
    "69": "Grapes",
    "70": "Christmas Trees",
    "71": "Other Tree Crops",
    "72": "Citrus",
    "74": "Pecans",
    "75": "Almonds",
    "76": "Walnuts",
    "77": "Pears",
    "81": "Clouds/No Data",
    "82": "Developed",
    "83": "Water",
    "87": "Wetlands",
    "88": "Nonag/Undefined",
    "92": "Aquaculture",
    "111": "Open Water",
    "112": "Perennial Ice/Snow",
    "121": "Developed/Open Space",
    "122": "Developed/Low Intensity",
    "123": "Developed/Med Intensity",
    "124": "Developed/High Intensity",
    "131": "Barren",
    "141": "Deciduous Forest",
    "142": "Evergreen Forest",
    "143": "Mixed Forest",
    "152": "Shrubland",
    "176": "Grassland/Pasture",
    "190": "Woody Wetlands",
    "195": "Herbaceous Wetlands",
    "204": "Pistachios",
    "205": "Triticale",
    "206": "Carrots",
    "207": "Asparagus",
    "208": "Garlic",
    "209": "Cantaloupes",
    "210": "Prunes",
    "211": "Olives",
    "212": "Oranges",
    "213": "Honeydew Melons",
    "214": "Broccoli",
    "215": "Avocados",
    "216": "Peppers",
    "217": "Pomegranates",
    "218": "Nectarines",
    "219": "Greens",
    "220": "Plums",
    "221": "Strawberries",
    "222": "Squash",
    "223": "Apricots",
    "224": "Vetch",
    "225": "Dbl Crop WinWht/Corn",
    "226": "Dbl Crop Oats/Corn",
    "227": "Lettuce",
    "228": "Dbl Crop Triticale/Corn",
    "229": "Pumpkins",
    "230": "Dbl Crop Lettuce/Durum Wht",
    "231": "Dbl Crop Lettuce/Cantaloupe",
    "232": "Dbl Crop Lettuce/Cotton",
    "233": "Dbl Crop Lettuce/Barley",
    "234": "Dbl Crop Durum Wht/Sorghum",
    "235": "Dbl Crop Barley/Sorghum",
    "236": "Dbl Crop WinWht/Sorghum",
    "237": "Dbl Crop Barley/Corn",
    "238": "Dbl Crop WinWht/Cotton",
    "239": "Dbl Crop Soybeans/Cotton",
    "240": "Dbl Crop Soybeans/Oats",
    "241": "Dbl Crop Corn/Soybeans",
    "242": "Blueberries",
    "243":"Cabbage",
    "244":"Cauliflower",
    "245":"Celery",
    "246":"Radishes",
    "247":"Turnips",
    "248":"Eggplants",
    "249":"Gourds",
    "250":"Cranberries",
    "254":"Dbl Crop Barley/Soybeans",
}


class CropYieldDFSummary:
    def __init__(self):

        self.df = pd.read_csv('/data2/hkaman/Data/CDL/Ultimate_Complete_Specialty_Crops_Data_with_Year.csv')
        self.df = self._preprocess_df()


    def _preprocess_df(self):
        """pre=processing includes three steps: 
            1. removing the crops with harvest area equal to 0
            2. removing the crops with yield 0 
            3. igonreo the data before 2017
            4. remove county called "State Total"
        
        """

        df = self.df[self.df['harvest_acres'] != 0]
        df = df[df['yield'] > 0]
        df = df[df['year'] >= 2017]
        df = df[self.df['county'] != "State Total"]
    
        return df


    def retunr_column_names(self):
        return self.df.columns
    
    def return_unique_list_var(self, var:str):

        return self.df[var].unique()


    def summary_top_common_crops(self):
        # Drop rows where county is "total state"
        self.df = self.df[self.df['county'] != "total state"]
        
        top_acres_per_county = {}
        top_yield_per_county = {}
        
        # Finding the top 20 crops for each county based on 'harvest_acres' and 'yield'
        for county in self.df['county'].unique():
            county_df = self.df[self.df['county'] == county]
            
            # Top 20 by harvest_acres
            top_acres = county_df.groupby('crop_name')['harvest_acres'].sum().nlargest(12)
            top_acres_per_county[county] = top_acres
            
            # Top 20 by yield
            top_yield = county_df.groupby('crop_name')['yield'].sum().nlargest(12)
            top_yield_per_county[county] = top_yield
        
        # Finding the most common crops across all counties
        all_top_acres = []
        all_top_yield = []
        
        for county, top_acres in top_acres_per_county.items():
            all_top_acres.extend(top_acres.index)
        
        for county, top_yield in top_yield_per_county.items():
            all_top_yield.extend(top_yield.index)
        
        # Counting the most common crops across all counties
        common_top_acres = Counter(all_top_acres).most_common(12)
        common_top_yield = Counter(all_top_yield).most_common(12)
        
        # Print the results
        # print("Top 20 crops by harvest_acres for each county:")
        # for county, crops in top_acres_per_county.items():
        #     print(f"{county}:")
        #     print(crops)
        #     print("\n")
        
        # print("Top 20 crops by yield for each county:")
        # for county, crops in top_yield_per_county.items():
        #     print(f"{county}:")
        #     print(crops)
        #     print("\n")
        
        print("Most common crops across all counties based on harvest_acres:")
        for crop, count in common_top_acres:
            print(f"{crop}: {count} counties")
        
        print("\nMost common crops across all counties based on yield:")
        for crop, count in common_top_yield:
            print(f"{crop}: {count} counties")

        # Returning both results
        # return top_acres_per_county, top_yield_per_county, common_top_acres, common_top_yield


    def summary_top_ten_crops(self, column='harvest_acres', county=None):
        # Check if the provided column is valid
        if column not in ['harvest_acres', 'yield']:
            raise ValueError("Column must be either 'harvest_area' or 'yield'")

        # Filter by county if provided
        if county:
            df = self.df[self.df['county'] == county]
        else:
            df = self.df

        # Calculate top 10 crops based on the selected column
        top_ten = df.groupby('crop_name')[column].sum().nlargest(10)
        
        # Print the results
        print(f"Top 10 crops by {column}:")
        for crop, value in top_ten.items():
            print(f"{crop}: {value}")

    #     # return top_ten


    def county_harvest_crops(self, county):
        """
        Returns a filtered dataframe for the specified county where harvest_acres > 0,
        along with a list of crop names with non-zero harvest_acres.
        """
        county_df = self.df[(self.df['county'] == county) & (self.df['harvest_acres'] > 0)]
        crop_names = county_df['crop_name'].unique().tolist()
        return county_df, crop_names

    def plot_timeseries_analysis(self, county=None, year_range=None, crops=None):
        """
        Creates 2x2 subplots for 'harvest_acres', 'yield', 'production', and 'value'.
        Plots separate lines for each crop if crops are specified.
        
        - county: Specify a county name. If None, all counties are considered.
        - year_range: Tuple (start_year, end_year). If None, all years are considered.
        - crops: Specify a crop name or a list of crop names. If None, all crops are considered.
        """
        df_filtered = self.df.copy()

        # Filter by county if provided
        if county:
            df_filtered = df_filtered[df_filtered['county'] == county]

        # Filter by year range if provided
        if year_range:
            start_year, end_year = year_range
            df_filtered = df_filtered[(df_filtered['year'] >= start_year) & (df_filtered['year'] <= end_year)]

        # Filter by specific crops if provided
        if crops:
            if isinstance(crops, str):
                crops = [crops]  # Convert single crop to list
            df_filtered = df_filtered[df_filtered['crop_name'].isin(crops)]

        # Columns to plot
        plot_columns = ['harvest_acres', 'yield', 'production', 'value']

        # Create 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        axs = axs.ravel()  # Flatten the 2x2 grid to 1D array for easier indexing

        # Plotting for each column
        for i, col in enumerate(plot_columns):
            ax = axs[i]
            if crops:
                # If specific crops are provided, plot each crop separately
                for crop in crops:
                    crop_df = df_filtered[df_filtered['crop_name'] == crop]
                    df_grouped = crop_df.groupby('year')[col].sum().reset_index()
                    ax.plot(df_grouped['year'], df_grouped[col], marker='o', label=crop)
                ax.legend()
            else:
                # If no crops are provided, plot the aggregate data
                df_grouped = df_filtered.groupby('year')[col].sum().reset_index()
                ax.plot(df_grouped['year'], df_grouped[col], marker='o')

            # Set title and labels for each subplot
            ax.set_title(f'{col.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel(col.replace("_", " ").title())
            ax.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_correlation_scatter(self, col_x, col_y, county=None, year_range=None, crops=None):
        """
        Calculates the correlation between two columns and plots a scatter plot.
        Optionally filters by county, year range, and crops.
        
        - col_x: The column to use for the x-axis.
        - col_y: The column to use for the y-axis.
        - county: Specify a county name. If None, all counties are considered.
        - year_range: Tuple (start_year, end_year). If None, all years are considered.
        - crops: Specify a crop name or a list of crop names. If None, all crops are considered.
        """
        df_filtered = self.df.copy()

        # Filter by county if provided
        if county:
            df_filtered = df_filtered[df_filtered['county'] == county]

        # Filter by year range if provided
        if year_range:
            start_year, end_year = year_range
            df_filtered = df_filtered[(df_filtered['year'] >= start_year) & (df_filtered['year'] <= end_year)]

        # Filter by specific crops if provided
        if crops:
            if isinstance(crops, str):
                crops = [crops]  # Convert single crop to list
            df_filtered = df_filtered[df_filtered['crop_name'].isin(crops)]

        # Calculate the correlation between the two columns
        correlation = df_filtered[[col_x, col_y]].corr().iloc[0, 1]

        # Scatter plot
        plt.figure(figsize=(14, 6))
        plt.scatter(df_filtered[col_x], df_filtered[col_y], alpha=0.7)
        plt.title(f'Scatter Plot of {col_x} vs {col_y}\nCorrelation: {correlation:.2f}')
        plt.xlabel(col_x.replace("_", " ").title())
        plt.ylabel(col_y.replace("_", " ").title())
        plt.grid(True)
        plt.show()

        # return correlation