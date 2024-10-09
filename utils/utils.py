import pandas as pd
import matplotlib.pyplot as plt

class CropYieldDFSummary:
    def __init__(self):

        self.df = pd.read_csv('/data2/hkaman/Data/CDL/Ultimate_Complete_Specialty_Crops_Data_with_Year.csv')


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