import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_load_data(data):
    # Use start time as plot reference value
    data['Start Time'] = data['Time (UTC)'].str.split(' - ').str[0]
    data['Start Time'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M')

    # Plot actual load vs entso-e forecast
    plt.figure(figsize=(12, 5))
    plt.plot(data['Start Time'], data['Actual Total Load [MW] - BZN|SE4'], label='Actual Load', color='blue')
    plt.plot(data['Start Time'], data['Day-ahead Total Load Forecast [MW] - BZN|SE4'], label='Forecasted load', color='red')
    plt.title('Electric Power Load - SE4')
    plt.xlabel('Time')
    plt.ylabel('Actual Total Load (MW)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def preprocess_load_data(file_path):
    df = pd.read_csv(file_path)

    # Keep original time range string
    df['Time (UTC)'] = df['Time (UTC)'].astype(str)

    # Create a processing timestamp from start of time range
    df['Timestamp'] = pd.to_datetime(df['Time (UTC)'].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')

    # Set timestamp as index
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Convert object dtypes to inferred types before interpolating
    df = df.infer_objects(copy=False)

    # Interpolate only numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

    # Group by timestamp, averaging only numeric data (ignoring strings)
    df_grouped = df.groupby(df.index)[numeric_cols].mean()

    # Reconstruct final DataFrame
    df_grouped = df_grouped.reset_index()
    df_grouped['Time (UTC)'] = df_grouped['Timestamp'].dt.strftime('%d.%m.%Y %H:00') + ' - ' + (
        df_grouped['Timestamp'] + pd.Timedelta(hours=1)).dt.strftime('%d.%m.%Y %H:00')
    df_grouped = df_grouped[['Time (UTC)'] + list(numeric_cols)]

    return df_grouped

totalLoadPath = os.path.join('LoadData','total_loadUTC_data.csv')
load_data = preprocess_load_data(totalLoadPath)
load_data.to_csv("processed_loadUTC_data.csv",index=False)
plot_load_data(load_data)

# print(load_data)
len_load = len(load_data)
print(len_load)

# Population data based on most recent readings
pop_malmo = 339316
pop_helsingborg = 152091
pop_kalmar = 72018
pop_halmstad = 106084
pop_vaxjo = 70489
pop_tot = pop_malmo + pop_helsingborg + pop_kalmar + pop_halmstad + pop_vaxjo
