import pandas as pd
import os

def get_weighted_average(data_dir, file_prefix, column_name, populations, output_filename):
    '''
    Computes the population-weighted average of different weather data types.

    Parameters:
        data_dir: Directory containing the input CSV files.
        file_prefix: Specifies prefix for specific weather data types.
        column_name: Column containing the weather for which the weighted average is computed.
        populations: Dictionary containing each city and their respective population.
        output_filename: Name of weigthed average csv file.

    Returns: DataFrame containing a weighted average time series.
    '''

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

populations = {'Malmo': 339316,
               'Helsingborg': 152091,
               'Kalmar': 72018,
               'Torup': 106084,
               'Vaxjo': 70489}

# Second set of population based on north and south
# Used for solar data where measurment locations are limited
population_area = {'Vaxjo': 248591,'Lund': 491407}

# Temperature
weighted_avg = get_weighted_average(data_dir='TemperatureData',
                                    file_prefix='Temp',
                                    column_name='Lufttemperatur',
                                    populations=populations,
                                    output_filename='weighted_avg_temp.csv')
print(weighted_avg.head())

# Humidity
weighted_avg = get_weighted_average(data_dir='HumidityData',
                                    file_prefix='HUMIDITY',
                                    column_name='Relativ Luftfuktighet',  
                                    populations=populations,
                                    output_filename='weighted_avg_humidity.csv')
print(weighted_avg.head())

# Solar
weighted_avg = get_weighted_average(data_dir='SolarData',
                                    file_prefix='SOLAR',
                                    column_name='Global Irradians (svenska stationer)',  
                                    populations=population_area,
                                    output_filename='weighted_avg_solar.csv')
print(weighted_avg.head())
