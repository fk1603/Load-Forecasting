import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Define the folder path where your CSV files are
#folder_path = r"C:\Users\andre\EG2140 - Computer applications and machine learning"

# Use glob to find all CSV files in that folder
# csv_files = sorted(glob.glob(os.path.join("LoadData", "Load*.csv")))

# Load and concatenate all CSV files
# df_list = [pd.read_csv(file, parse_dates=['Time (CET/CEST)']) for file in csv_files]
# merged_df = pd.concat(df_list)


# Save the merged dataframe to a new CSV file
# output_path = os.path.join("LoadData", "total_load_data.csv")
# merged_df.to_csv(output_path, index=False)

# print(f"Merged CSV saved to: {output_path}")

# Define the relative folder path from your current directory (Load-Forecasting)
#folder_path = "LoadData"

# Use glob to find all CSV files that match the pattern in the LoadData folder
#csv_files = sorted(glob.glob(os.path.join(folder_path, "LoadUTC*.csv")))

# Load and concatenate all CSV files
#df_list = [pd.read_csv(file, parse_dates=['Time (UTC)']) for file in csv_files]
#merged_df = pd.concat(df_list)

# Save the merged dataframe to a new CSV file
#output_path = os.path.join(folder_path, "total_loadUTC_data.csv")
#merged_df.to_csv(output_path, index=False)

#print(f"Merged CSV saved to: {output_path}")


def get_weighted_average(data_dir, file_prefix, column_name, populations, output_filename):
    dfs = {}
    base_dir = os.getcwd()
    for city in populations:
        file_name = f'{file_prefix}{city}_processed.csv'
        file_path = os.path.join(base_dir, data_dir, file_name)
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        dfs[city] = df

    # Combine data
    combined = pd.DataFrame(index=dfs[list(populations.keys())[0]].index)
    for city, df in dfs.items():
        combined[city] = df[column_name] * populations[city]

    total_population = sum(populations.values())
    combined['weighted_avg'] = combined.sum(axis=1) / total_population

    weighted_avg = combined[['weighted_avg']]
    weighted_avg.to_csv(output_filename)

    return weighted_avg

populations = {
    'Malmo': 339316,
    'Helsingborg': 152091,
    'Kalmar': 72018,
    'Torup': 106084,
    'Vaxjo': 70489
}

population_area = {
    'Vaxjo': 248591,
    'Lund': 491407
}

# Temperature
weighted_avg = get_weighted_average(
    data_dir='TemperatureData',
    file_prefix='Temp',
    column_name='Lufttemperatur',
    populations=populations,
    output_filename='weighted_avg_temp.csv'
)
print(weighted_avg.head())

# Humidity
weighted_avg = get_weighted_average(
    data_dir='HumidityData',
    file_prefix='HUMIDITY',
    column_name='Relativ Luftfuktighet',  
    populations=populations,
    output_filename='weighted_avg_humidity.csv'
)
print(weighted_avg.head())

# Solar
weighted_avg = get_weighted_average(
    data_dir='SolarData',
    file_prefix='SOLAR',
    column_name='Global Irradians (svenska stationer)',  
    populations=population_area,
    output_filename='weighted_avg_solar.csv'
)
print(weighted_avg.head())
