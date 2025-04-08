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
    temp_data = pd.read_csv(
        file,
        sep=';',
        skiprows=start_idx,
        names=['Datum', 'Tid (UTC)', 'Lufttemperatur', 'Kvalitet'],
        usecols=[0, 1, 2, 3],
        encoding='utf-8'
    )

    # Combine date and time into a datetime column
    temp_data['Timestamp'] = pd.to_datetime(temp_data['Datum'] + ' ' + temp_data['Tid (UTC)'])
    return temp_data

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
# plot_load_data(load_data)

load_data = pd.read_csv('Load2015.csv')
# plot_load_data(load_data)

# temp_data = pd.read_csv('TempMalmoA.csv')
temp_data = process_temp_data('TempVaxsjo.csv')
plot_temp_data(temp_data)