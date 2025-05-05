#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import missingno as mno
import pickle
import warnings
warnings.filterwarnings("ignore")
import logging
import torch
torch.set_float32_matmul_precision('medium')
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.models import NHiTSModel
from darts.models import TransformerModel
from darts.models import NBEATSModel
from darts.metrics import mape, rmse, mse, mae, r2_score,smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback, EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)
from darts.metrics import mape, rmse, mse, mae, r2_score,smape
from optuna.samplers import TPESampler

pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format
# load
df0 = pd.read_csv("processed_load_data.csv", header=0, parse_dates=["Time (CET/CEST)"])
df0 = df0.rename(columns={"Time (CET/CEST)": "Timestamp"})
df0.columns = ["Timestamp", "Forecast", "Actual_Load"]
#df0 contains Timestamp and Actual Load Consumption
df0 = df0.drop(columns=["Forecast"])  # Drop 'Forecast'

dfw0 = pd.read_csv("weighted_avg_humidity.csv", header=0, parse_dates=["Timestamp"])
dfw0.columns = ["Timestamp", "Humidity"]
dfw1 = pd.read_csv("weighted_avg_solar.csv", header=0, parse_dates=["Timestamp"])
dfw1.columns = ["Timestamp", "Solar"]
dfw2 = pd.read_csv("weighted_avg_temp.csv", header=0, parse_dates=["Timestamp"])
dfw2.columns = ["Timestamp", "Temperature"]
dfw0 = dfw0.merge(dfw1, on="Timestamp").merge(dfw2, on="Timestamp")
# Convert df0["Timestamp"] to datetime by extracting the first timestamp
df0["Timestamp"] = df0["Timestamp"].str.extract(r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})")[0]
df0["Timestamp"] = pd.to_datetime(df0["Timestamp"], format="%d.%m.%Y %H:%M")
# Ensure "Actual" is treated as a string
df0["Actual_Load"] = df0["Actual_Load"].astype(str).str.replace(",", "").astype(float)
# Now merge
df_merged = df0.merge(dfw0, on="Timestamp")
# Remove rows between 2015 and 2020 (inclusive)
df_merged = df_merged[~df_merged["Timestamp"].dt.year.between(2015, 2020)]

# Print the resulting DataFrame
print(df_merged)
# backup of original sources
import holidays
from pandas.tseries.offsets import CustomBusinessDay
# Define Swedish holidays using the `holidays` package
swe_holidays = holidays.Sweden(years=range(2020
                                           , 2024))
# Create a CustomBusinessDay offset excluding Swedish holidays
sweden_busday = CustomBusinessDay(holidays=swe_holidays.keys())
# Example DataFrame with a 'DateTime' column
# Add Holiday column: 1 if date is a holiday, 0 otherwise
df_merged['Holiday'] = df_merged['Timestamp'].dt.date.apply(lambda x: 1 if x in swe_holidays else 0)
# convert int and float64 columns to float32
intcols = list(df_merged.dtypes[df_merged.dtypes == np.int64].index)
df_merged[intcols] = df_merged[intcols].applymap(np.float32)

# 4. Pearson correlation: 'Actual' vs. each weather variable
correlation_matrix = df_merged.corr(method="pearson")
# Show correlation of 'Actual' with others
print(correlation_matrix)
# Plot heatmap
# Assuming df_merged contains only numerical columns for correlation
numerical_df = df_merged.drop(columns=["Timestamp"])
# Compute correlation matrix
corr = numerical_df.corr()
# Plot heatmap with labels but without grid lines
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0, cbar_kws={'shrink': 0.75}, square=True)

# Remove the gridlines (between the boxes)
plt.grid(False)

plt.title("Correlation Heatmap")
#plt.show()

# First extract Weekday, Month, and Hour
df_merged["Weekday"] = df_merged["Timestamp"].dt.weekday  # 0 = Monday
df_merged["Month"] = df_merged["Timestamp"].dt.month
df_merged["Hour"] = df_merged["Timestamp"].dt.hour

# 1. Weekday vs. Month Heatmap of mean Actual_Load
weekday_month_data = df_merged.pivot_table(index='Weekday', columns='Month', values='Actual_Load', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(weekday_month_data, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0, cbar_kws={'shrink': 0.75}, square=True)
plt.title("Mean Load by Weekday and Month")
plt.ylabel("Weekday (0=Mon)")
plt.xlabel("Month")
# Remove the gridlines (between the boxes)
plt.grid(False)
#plt.show()

# 2. Weekday vs. Hour Heatmap of mean Actual_Load
weekday_hour_data = df_merged.pivot_table(index='Weekday', columns='Hour', values='Actual_Load', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(weekday_hour_data, annot=False, cmap="coolwarm", fmt=".2f",
            linewidths=0, cbar_kws={'shrink': 0.75}, square=True)
plt.title("Mean Load by Weekday and Hour")
plt.ylabel("Weekday (0=Mon)")
plt.xlabel("Hour")
# Remove the gridlines (between the boxes)
plt.grid(False)
#plt.show()

#create timeseries object for target variable i.e Actual_Load
# Assuming 'Timestamp' column exists and represents the time
ts_P = TimeSeries.from_dataframe(df_merged, time_col="Timestamp", value_cols="Actual_Load")
# check attributes of the time series
print("components:", ts_P.components)                   # List of all component names (e.g. ['Actual_Load', 'Humidity', ...])
print("duration:", ts_P.duration)                       # Total timedelta between first and last entry
print("frequency:", ts_P.freq)                          # Pandas offset alias for frequency (e.g. <Minute>)
print("frequency:", ts_P.freq_str)                      # String version of frequency (e.g. "15min", "1H")
print("has date time index?", ts_P.has_datetime_index)  # True if the index is datetime-based
print("deterministic:", ts_P.is_deterministic)          # True if the series is deterministic
print("univariate:", ts_P.is_univariate)                # True if the series has only one component
# create time series object for the feature columns
# Set the Timestamp as the index
df_covF = df_merged.loc[:, df_merged.columns != "Actual_Load"]
df_covF.set_index('Timestamp', inplace=True)

# Now, create the time series for feature columns
ts_covF = TimeSeries.from_dataframe(df_covF)

# Check the attributes
print("components (columns) of feature time series:", ts_covF.components)
print("duration:", ts_covF.duration)
print("frequency:", ts_covF.freq)
print("frequency:", ts_covF.freq_str)
print("has date time index? (or else, it must have an integer index):", ts_covF.has_datetime_index)
print("deterministic:", ts_covF.is_deterministic)
print("univariate:", ts_covF.is_univariate)

# example: operating with time series objects:
# we can also create a 3-dimensional numpy array from a time series object
# 3 dimensions: time (rows) / components (columns) / samples
ar_covF = ts_covF.all_values()
print(type(ar_covF))

# example: operating with time series objects:
# we can also create a pandas series or dataframe from a time series object
df_covF = ts_covF.to_dataframe()
type(df_covF)

SPLIT_TRAIN = 0.6
SPLIT_VAL = 0.2
SPLIT_TEST = 0.2

from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

# 1. Train/val/test split
train_size = int(len(ts_P) * SPLIT_TRAIN)
val_size = int(len(ts_P) * SPLIT_VAL)
test_size = len(ts_P) - train_size - val_size

ts_train, ts_val, ts_test = ts_P[:train_size], ts_P[train_size:train_size+val_size], ts_P[-test_size:]
cov_train, cov_val, cov_test = ts_covF[:train_size], ts_covF[train_size:train_size+val_size], ts_covF[-test_size:]

# 2. Normalize target and features
scaler_target = Scaler()
scaler_cov = Scaler()

ts_train_scaled = scaler_target.fit_transform(ts_train)
ts_val_scaled = scaler_target.transform(ts_val)
ts_test_scaled = scaler_target.transform(ts_test)

cov_train_scaled = scaler_cov.fit_transform(cov_train)
cov_val_scaled = scaler_cov.transform(cov_val)
cov_test_scaled = scaler_cov.transform(cov_test)

# 3. Instantiate and train TFT
input_chunk_length = 168  # 1 week of hourly data
forecast_horizon = 48     # next 48 hours
output_chunk_length = 48

from darts.models import TFTModel

from darts.models import TFTModel

tft = TFTModel(
    input_chunk_length=24,
    output_chunk_length=12,
    hidden_size=16,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=100,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},
    add_encoders={
        "future": ["hour", "dayofweek", "month", "relative"],
        "cyclic": ["hour", "dayofweek", "month"]
    },
    force_reset=True
)


tft.fit(
    series=ts_train_scaled,
    past_covariates=cov_train_scaled,
    val_series=ts_val_scaled,
    val_past_covariates=cov_val_scaled,
    verbose=True
)

# 4. Forecast the next 48 hours using the latest available input window
forecast = tft.predict(
    n=forecast_horizon,
    series=ts_train_scaled.append(ts_val_scaled),
    past_covariates=cov_train_scaled.append(cov_val_scaled)
)

# 5. Inverse transform and plot
forecast_orig = scaler_target.inverse_transform(forecast)

ts_actual = ts_test[:forecast_horizon]
ts_actual = ts_actual[:len(forecast_orig)]  # align if not exact length

# Plot
ts_actual.plot(label="Actual")
forecast_orig.plot(label="TFT Forecast (Median)")
plt.title("TFT 48-Hour Forecast vs Actual")
plt.legend()
plt.show()
