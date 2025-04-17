import pandas as pd
from darts import TimeSeries
import darts.utils.timeseries_generation as tg
import matplotlib.pyplot as plt

# Step 1: Load your CSV data
df = pd.read_csv('processed_load_data.csv')

# Print the column names to inspect the CSV file structure
print(df.columns)

# Step 2: Optionally, inspect the first few rows of the DataFrame to check the data
print(df.head())

# Step 3: If 'timestamp' exists in the columns, convert it to datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 4: Plot the data (assuming 'timestamp' and 'value' are the column names)
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['value'], label='Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Plot')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
