import pandas as pd
import glob
import os

# Define load data file path
base_dir = os.getcwd()
folder_path = os.path.join(base_dir, "LoadData")

# Use glob to find all csv files that match the pattern in the directory
csv_files = sorted(glob.glob(os.path.join(folder_path, "LoadUTC*.csv")))

# Load and concatenate all csv files
df_list = [pd.read_csv(file, parse_dates=['Time (UTC)']) for file in csv_files]
merged_df = pd.concat(df_list)

# Save the merged dataframe to a new csv file
output_path = os.path.join(folder_path, "total_loadUTC_data.csv")
merged_df.to_csv(output_path, index=False)

print(f"Merged CSV saved to: {output_path}")