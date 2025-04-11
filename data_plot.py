import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_load_data(data):
    # Use start time as plot reference value
    data['Start Time'] = data['Time (CET/CEST)'].str.split(' - ').str[0]
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

def process_temp_data(file):
    # Skip metadata rows and find the actual data block
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the start of actual measurements (line that starts with 'Datum')
    for i, line in enumerate(lines):
        if line.strip().startswith('Datum'):
            start_idx = i + 1
            break

    # Read actual measurement data into DataFrame
    temp_data = pd.read_csv(file,
                            sep=';',
                            skiprows=start_idx,
                            names=['Datum', 'Tid (UTC)', 'Lufttemperatur', 'Kvalitet'],
                            usecols=[0, 1, 2, 3],
                            encoding='utf-8')

    # Combine date and time into a datetime column
    temp_data['Timestamp'] = pd.to_datetime(temp_data['Datum'] + ' ' + temp_data['Tid (UTC)'])
    return temp_data

def fill_missing_timestamps(temp_df, freq='1h'):
    """
    Fill missing timestamps in a temperature DataFrame.
    
    Parameters:
        temp_df (DataFrame): Must include a 'Timestamp' column.
        freq (str): Expected frequency of the time series (e.g., '1H' for hourly).
    
    Returns:
        DataFrame with a complete timestamp index and NaNs where data is missing.
    """
    # Set timestamp as index
    temp_df = temp_df.set_index('Timestamp')
    
    # Sort index to ensure correct order
    temp_df = temp_df.sort_index()

    # Create a full timestamp range from start to end
    full_index = pd.date_range(start=temp_df.index.min(), end=temp_df.index.max(), freq=freq)

    # Reindex to fill missing timestamps
    temp_df = temp_df.reindex(full_index)

    # Rename index back to 'Timestamp' for consistency
    temp_df.index.name = 'Timestamp'
    temp_df = temp_df.reset_index()

    return temp_df

def plot_temp_data(data):
    # Plot actual load vs entso-e forecast
    plt.figure(figsize=(12, 5))
    plt.plot(data['Timestamp'], data['Lufttemperatur'], label='Temperature (°C)', color='green')
    plt.title('Temperature Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

load_data = pd.read_csv('total_load_data.csv')
plot_load_data(load_data)
print(load_data)

load_data_1yr = pd.read_csv('Load2015.csv')
len_load_daya_1yr = len(load_data_1yr)
print('1yr data')
print(len_load_daya_1yr)

temp_data = process_temp_data('TempVaxsjo.csv')
temp_data = fill_missing_timestamps(temp_data, freq='1h')
plot_temp_data(temp_data)
print(temp_data)
# load_data['Tempreature'] = temp_data['Lufttemperatur']

# print(load_data)
len_load = len(load_data)
print(len_load)
len_temp = len(temp_data)
print(len_temp)

temp_malmo = process_temp_data('TempMalmoA.csv')
len_temp_malmo = len(temp_malmo)
print(len_temp_malmo)

temp_helsingborg = process_temp_data('TempHelsingborgA.csv')
len_temp_helsingborg = len(temp_helsingborg)
print(len_temp_helsingborg)

temp_kalmar = process_temp_data('TempKalmar.csv')
len_temp_kalmar = len(temp_kalmar)
print(len_temp_kalmar)

temp_halmstad = process_temp_data('TempTorup.csv')
len_temp_halmstad = len(temp_halmstad)
print(len_temp)

temp_vaxjo = process_temp_data('TempVaxsjo.csv')
len_temp_vaxjo = len(temp_vaxjo)
print(len_temp_vaxjo)

# Population data based on most recent readings
pop_malmo = 339316
pop_helsingborg = 152091
pop_kalmar = 72018
pop_halmstad = 106084
pop_vaxjo = 70489
pop_tot = pop_malmo + pop_helsingborg + pop_kalmar + pop_halmstad + pop_vaxjo
