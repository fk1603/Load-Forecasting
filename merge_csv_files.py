import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Define the folder path where your CSV files are
folder_path = r"C:\Users\andre\EG2140 - Computer applications and machine learning"

# Use glob to find all CSV files in that folder
csv_files = sorted(glob.glob(os.path.join(folder_path, "Load*.csv")))

# Load and concatenate all CSV files
df_list = [pd.read_csv(file, parse_dates=['Time (CET/CEST)']) for file in csv_files]
merged_df = pd.concat(df_list)


# Save the merged dataframe to a new CSV file
output_path = os.path.join(folder_path, "total_load_data.csv")
merged_df.to_csv(output_path, index=False)

# print(f"Merged CSV saved to: {output_path}")

