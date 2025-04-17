import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_temp_data(file):
    # Skip metadata rows and find the actual header row (starts with 'Datum')
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    #header_idx = None
    # Find the header row index
    for i, line in enumerate(lines):
        if line.strip().startswith('Datum'):
            header_idx = i
            break

    # Read the file starting from the header line
    temp_data = pd.read_csv(
        file,
        sep=';',
        header=0,
        skiprows=header_idx,  # This will treat the correct line as header
        encoding='utf-8'
    )

    # Combine date and time columns into a single datetime column (best guess on column names)
    date_col = [col for col in temp_data.columns if 'Datum' in col][0]
    time_col = [col for col in temp_data.columns if 'Tid' in col][0]

    temp_data['Timestamp'] = pd.to_datetime(temp_data[date_col] + ' ' + temp_data[time_col])

    return temp_data

def fill_missing_timestamps(temp_data, freq='1h'):
    # Ensure 'Datum' and 'Tid (UTC)' columns exist
    if 'Datum' not in temp_data.columns or 'Tid (UTC)' not in temp_data.columns:
        raise ValueError("The input DataFrame must contain 'Datum' and 'Tid (UTC)' columns.")
    
    # Combine Date and Time into a single datetime column
    if 'Timestamp' not in temp_data.columns:
        temp_data['Timestamp'] = pd.to_datetime(temp_data['Datum'] + ' ' + temp_data['Tid (UTC)'])
    
    # Set 'Timestamp' as index
    temp_data.set_index('Timestamp', inplace=True)

    # Sort index to ensure correct order
    temp_data = temp_data.sort_index()

    # Create a full timestamp range from start to end
    full_index = pd.date_range(start=temp_data.index.min(), end=temp_data.index.max(), freq=freq)

    # Reindex to fill missing timestamps
    temp_data = temp_data.reindex(full_index)

    # Interpolate missing values in 'Lufttemperatur'
    if 'Lufttemperatur' in temp_data.columns:
        temp_data['Lufttemperatur'] = temp_data['Lufttemperatur'].interpolate(method='linear')
    elif 'Relativ Luftfuktighet' in temp_data.columns:
        temp_data['Relativ Luftfuktighet'] = temp_data['Relativ Luftfuktighet'].interpolate(method='linear')
    elif 'Global Irradians (svenska stationer)' in temp_data.columns:
        temp_data['Global Irradians (svenska stationer)'] = temp_data['Global Irradians (svenska stationer)'].interpolate(method='linear')
  
    # Rename index back to 'Timestamp' for consistency
    temp_data.index.name = 'Timestamp'
    
    # Reset index if you want Timestamp as a column instead of index
    temp_data.reset_index(inplace=True)

    return temp_data

# List of file names and labels for plotting
files = [
    ('TempMalmo.csv', 'Malmo', 'TemperatureData'),
    ('TempHelsingborg.csv', 'Helsingborg', 'TemperatureData'),
    ('TempKalmar.csv', 'Kalmar', 'TemperatureData'),        
    ('TempTorup.csv', 'Halmstad', 'TemperatureData'),
    ('TempVaxjo.csv', 'Vaxjo', 'TemperatureData'),
    ('HUMIDITYVaxjo.csv', 'Vaxjo', 'HumidityData'),
    ('HUMIDITYKalmar.csv', 'Kalmar', 'HumidityData'),
    ('HUMIDITYMalmo.csv', 'Malmo', 'HumidityData'),
    ('HUMIDITYTorup.csv', 'Torup', 'HumidityData'),
    ('HUMIDITYHelsingborg.csv', 'Helsingborg', 'HumidityData'),
    ('SOLARLund.csv', 'Lund', 'SolarData'),
    ('SOLARVaxjo.csv', 'Vaxjo', 'SolarData')
]

# Loop through each file to process and plot
for filename, label, dataType in files:
    # Process the data
    filename = os.path.join(dataType,filename)
    temp_data = process_temp_data(filename)
    temp_data_filled = fill_missing_timestamps(temp_data, freq='1h')
    
    # Save the processed DataFrame to a new CSV
    processed_filename = filename.replace('.csv', '_processed.csv')
    temp_data_filled.to_csv(processed_filename, index=False)
    
    # Print the length of the processed DataFrame
    print(f"{label} - Number of entries: {len(temp_data_filled)}")

    # Create a new figure for each dataset
    plt.figure(figsize=(12, 6))
    
    # Plot the temperature data
    if 'Lufttemperatur' in temp_data.columns:
        plt.plot(temp_data_filled['Timestamp'], temp_data_filled['Lufttemperatur'], label=label, color='blue')
    
        plt.title(f'Temperature Data for {label}')
        plt.ylabel('Temperature (°C)')
        plt.xlabel('Timestamp')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_name = os.path.join(dataType, f"{label}temperature.png")
        plt.savefig(plot_name)
    
    elif 'Relativ Luftfuktighet' in temp_data.columns:
        plt.plot(temp_data_filled['Timestamp'], temp_data_filled['Relativ Luftfuktighet'], label=label, color='blue')
    
        plt.title(f'Humidity Data for {label}')
        plt.ylabel('Humidity (%)')
        plt.xlabel('Timestamp')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_name = os.path.join(dataType, f"{label}_humidity.png")
        plt.savefig(plot_name)
    
    elif 'Global Irradians (svenska stationer)' in temp_data.columns:
        plt.plot(temp_data_filled['Timestamp'], temp_data_filled['Global Irradians (svenska stationer)'], label=label, color='blue')
    
        plt.title(f'Sun Irradiation for {label}')
        plt.ylabel('Sun irradiation (W/m^2)')
        plt.xlabel('Timestamp')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_name = os.path.join(dataType, f"{label}_sunirradiation.png")
        plt.savefig(plot_name)

    plt.close()  # Close the plot to free memory

print("All plots saved successfully.")