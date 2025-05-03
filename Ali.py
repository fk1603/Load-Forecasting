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
df0 = pd.read_csv("processed_loadUTC_data.csv", header=0, parse_dates=["Time (UTC)"])
df0 = df0.rename(columns={"Time (UTC)": "Timestamp"})
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
swe_holidays = holidays.Sweden(years=range(2021, 2025))
# Create a CustomBusinessDay offset excluding Swedish holidays
sweden_busday = CustomBusinessDay(holidays=swe_holidays.keys())
# Create a date range for the entire period
date_range = pd.date_range(start='2021-01-01', end='2024-12-31', freq=sweden_busday)
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
numerical_df = df_merged.drop(columns=["Timestamp"])
# Compute correlation matrix
corr = numerical_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0, cbar_kws={'shrink': 0.75}, square=True)
plt.grid(False)
plt.title("Correlation Heatmap")
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

SPLIT_TRAIN = 0.5
SPLIT_VAL = 0.25
SPLIT_TEST = 0.25
# Calculate indices for splitting
train_idx = int(len(ts_P) * SPLIT_TRAIN)
val_idx = int(len(ts_P) * (SPLIT_TRAIN + SPLIT_VAL))

# Train/val/test split
ts_train = ts_P[:train_idx]
ts_val = ts_P[train_idx:val_idx]
ts_test = ts_P[val_idx:]

print("training start:", ts_train.start_time())
print("training end:", ts_train.end_time())
print("training duration:",ts_train.duration)
print("Validation start:", ts_val.start_time())
print("Validation end:", ts_val.end_time())
print("Validation duration:", ts_val.duration)
print("test start:", ts_test.start_time())
print("test end:", ts_test.end_time())
print("test duration:", ts_test.duration)


scalerP = Scaler()
scalerP.fit(ts_train)
ts_ttrain = scalerP.transform(ts_train)
ts_tval = scalerP.transform(ts_val)
ts_ttest = scalerP.transform(ts_test)
ts_t = ts_P

# make sure data are of type float
ts_t = ts_t.astype("float32")
ts_ttrain = ts_ttrain.astype("float32")
ts_tval = ts_tval.astype("float32")
ts_ttest = ts_ttest.astype("float32")

# Calculate indices for splitting
train_idx = int(len(ts_covF) * SPLIT_TRAIN)
val_idx = int(len(ts_covF) * (SPLIT_TRAIN + SPLIT_VAL))

# Train/val/test split for feature covariates
covF_train = ts_covF[:train_idx]
covF_val = ts_covF[train_idx:val_idx]
covF_test = ts_covF[val_idx:]

# Scale feature covariates
scalerF = Scaler()
scalerF.fit_transform(covF_train)
covF_ttrain = scalerF.transform(covF_train)
covF_tval = scalerF.transform(covF_val)
covF_ttest = scalerF.transform(covF_test)
covF_t = scalerF.transform(ts_covF)

# Make sure data are of type float
covF_ttrain = covF_ttrain.astype(np.float32)
covF_tval = covF_tval.astype(np.float32)
covF_ttest = covF_ttest.astype(np.float32)

pd.options.display.float_format = '{:.2f}'.format
print("first and last row of scaled feature covariates:")

# Feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
covT = datetime_attribute_timeseries(ts_P.time_index, attribute="hour")
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="day_of_week"))
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="month"))
covT = covT.stack(datetime_attribute_timeseries(covT.time_index, attribute="year"))

# Train/val/test split for time covariates
covT_train = covT[:train_idx]
covT_val = covT[train_idx:val_idx]
covT_test = covT[val_idx:]

covT_val_extended = covT_val.to_dataframe()
additional_time_index = pd.date_range(start=covT_val_extended.index[-1] + pd.Timedelta(hours=1),
                                      periods=48, freq='H')
additional_rows = pd.DataFrame(index=additional_time_index, columns=covT_val_extended.columns)
covT_val_extended = pd.concat([covT_val_extended, additional_rows], ignore_index=False)
covT_val_extended = TimeSeries.from_dataframe(covT_val_extended)

# Rescale the covariates: fitting on the training set
scalerT = Scaler()
scalerT.fit(covT_train)
covT_ttrain = scalerT.transform(covT_train)
covT_tval = scalerT.transform(covT_val)  # Scale before extending
covT_ttest = scalerT.transform(covT_test)
covT_t = scalerT.transform(covT)

covT_t = covT_t.astype(np.float32)
covT_ttrain = covT_ttrain.astype(np.float32)
covT_tval = covT_tval.astype(np.float32)
covT_ttest = covT_ttest.astype(np.float32)

pd.options.display.float_format = '{:.0f}'.format
print("first and last row of unscaled time covariates:")

# Combine feature and time covariates along component dimension: axis=1
ts_cov = ts_covF.concatenate(covT.slice_intersect(ts_covF), axis=1)  # unscaled F+T
cov_t = covF_t.concatenate(covT_t.slice_intersect(covF_t), axis=1)  # scaled F+T
cov_ttrain = covF_ttrain.concatenate(covT_ttrain.slice_intersect(covF_ttrain), axis=1)  # scaled F+T training set
cov_tval = covF_tval.concatenate(covT_tval.slice_intersect(covF_tval), axis=1)  # scaled F+T validation set
cov_ttest = covF_ttest.concatenate(covT_ttest.slice_intersect(covF_ttest), axis=1)  # scaled F+T test set

pd.options.display.float_format = '{:.2f}'.format
print("first and last row of unscaled covariates:")

print("covF_ttrain start:", covF_ttrain.start_time())
print("covF_ttrain end:", covF_ttrain.end_time())
print("covF_tval start:", covF_tval.start_time())
print("covF_tval end:", covF_tval.end_time())
print("covF_ttest start:", covF_ttest.start_time())
print("covF_ttest end:", covF_ttest.end_time())

print("covT_ttrain start:", covT_ttrain.start_time())
print("covT_ttrain end:", covT_ttrain.end_time())
print("covT_tval start:", covT_tval.start_time())
print("covT_tval end:", covT_tval.end_time())
print("covT_ttest start:", covT_ttest.start_time())
print("covT_ttest end:", covT_ttest.end_time())

LOAD = False# True = load previously saved model from disk? False = (re)train the model

SAVE = "_TFT_model_single_step.pt"  # file name to save the model under
mpath = r'c:\\Users\\Ali Kamran\\Downloads\\model' + SAVE  # file path where the model will be saved or loaded from


def build_fit_tft_model(
        input_chunk_length,
        output_chunk_length,
        hidden_size,
        lstm_layers,
        num_attention_heads,
        dropout,
        lr,
        callbacks=None,

):
    # reproducibility
    torch.manual_seed(42)

    batch_size = 256
    n_epochs = 70
    nr_epochs_val_period = 1
    QUANTILES = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]

    # Monitor validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.005, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    # Build the TFT model
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        batch_size=batch_size,
        n_epochs=n_epochs,
        nr_epochs_val_period=nr_epochs_val_period,
        likelihood=QuantileRegression(QUANTILES),
        model_name="tft_model_1h",
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model is using device: {device}")

    # Train the model
    # Ensure all inputs are float32
    model.fit(
        series=ts_ttrain.astype(np.float32),
        future_covariates=cov_t.astype(np.float32),
        val_series=ts_tval.astype(np.float32),
        val_future_covariates=cov_t.astype(np.float32),
        verbose=True,
    )

    return model
QUANTILES =[
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback

class CustomPyTorchLightningPruningCallback(Callback):
    def __init__(self, trial, monitor="val_loss"):
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=trainer.global_step)
        if self._trial.should_prune():
            message = f"Trial was pruned at step {trainer.global_step}."
            raise optuna.TrialPruned(message)

def objective(trial):
    callback = [CustomPyTorchLightningPruningCallback(trial, monitor="val_loss")]

    input_chunk_length = trial.suggest_categorical("input_chunk_length", [12, 24, 48])
    hidden_size = trial.suggest_categorical("hidden_size", [12, 24, 32])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    num_attention_heads = trial.suggest_int("num_attention_heads", 2, 6)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log = True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # build and train the TFT model with these hyper-parameters:
    tft_model = build_fit_tft_model(
        input_chunk_length=input_chunk_length,
        output_chunk_length = 24,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        lr=lr,
        dropout=dropout,
        callbacks=callback
    )


    # Evaluate how good it is on the validation set
    #In the next trial, increase the num_samples MAPE is signifantly dependant on n_samples
    preds = tft_model.predict(n=len(ts_tval), num_samples=1, n_jobs=os.cpu_count(), verbose=True)
    print('0 Ali')
    error = mape(preds, ts_tval)

    return error if error != np.nan else float("inf")
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

# Ali.py
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# Select the target column
target_col = 'Actual_Load'
df2_1 = df_merged.copy()

df2_numeric = df_merged.select_dtypes(include=[np.number])  # This includes only numeric columns

# Convert target column to numeric if it's not already
df_merged[target_col] = pd.to_numeric(df_merged[target_col], errors='coerce')

# Prepare the features and target data
X = df2_numeric.drop(columns=[target_col])
y = df2_numeric[target_col]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the LGBMRegressor model
lgbm_model = LGBMRegressor()

# Perform RFE using the LGBMRegressor model
rfe = RFE(estimator=lgbm_model, n_features_to_select=7, step=1)
rfe.fit(X_train_scaled, y_train)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Print the selected features
print("Selected features:")
print(selected_features)

# Create a new dataframe with the selected features and target
selected_df = df2_numeric[selected_features.to_list() + [target_col]]

# Train and SAVE the TFT model using the best hyperparameters found by Optuna
sampler = TPESampler(seed=42)  # Define sampler here

if LOAD:
    try:
        print("have loaded a previously saved model from disk:" + mpath)
        best_model = TFTModel.load(mpath)
        # Add code here to evaluate or use the loaded model if needed
    except FileNotFoundError:
        print(f"Warning: Model not found at {mpath}. Training a new model using best Optuna parameters.")
        study = optuna.create_study(study_name="T14", storage="sqlite:///optimization.db", load_if_exists=True,
                                    sampler=sampler, direction="minimize")
        best_params = study.best_trial.params
        best_model = build_fit_tft_model(
            input_chunk_length=best_params['input_chunk_length'],
            output_chunk_length=1,
            hidden_size=best_params['hidden_size'],
            lstm_layers=best_params['lstm_layers'],
            num_attention_heads=best_params['num_attention_heads'],
            lr=best_params['lr'],
            dropout=best_params['dropout'],
        )
        print("have saved the best model after training with Optuna's best hyperparameters:", mpath)
        best_model.save(mpath)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Handle other potential loading errors
else:
    # Code to run if LOAD is False (force retraining with Optuna)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(study_name="T14", storage="sqlite:///optimization.db", load_if_exists=True, sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=300, callbacks=[print_callback])
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
    best_params = study.best_trial.params
    best_model = build_fit_tft_model(
        input_chunk_length=best_params['input_chunk_length'],
        output_chunk_length=24,
        hidden_size=best_params['hidden_size'],
        lstm_layers=best_params['lstm_layers'],
        num_attention_heads=best_params['num_attention_heads'],
        lr=best_params['lr'],
        dropout=best_params['dropout'],
    )
    print("have saved the best model after training with Optuna's best hyperparameters:", mpath)
    best_model.save(mpath)


# Try a reduced number of samples for faster evaluation
num_eval_samples = 500  # Adjust this number as needed

# Experiment with the number of parallel jobs
n_prediction_jobs = os.cpu_count()  # Or try a slightly smaller value

tft_predictions = best_model.predict(
    n=len(ts_tval),
    num_samples=num_eval_samples,
    n_jobs=n_prediction_jobs,
    verbose=True
)
error = mape(tft_predictions, ts_tval)
print(error)
# retrieve forecast series for chosen quantiles,
# inverse-transform each series,
# insert them as columns in a new dataframe dfY
q50_RMSE = np.inf
q50_MAPE = np.inf
ts_q50 = None
pd.options.display.float_format = '{:,.2f}'.format
dfY_tft = pd.DataFrame()
dfY_tft["Actual"] = TimeSeries.to_series(ts_val)


# helper function: get forecast values for selected quantile q and insert them in dataframe dfY
def predQ(ts_t, q):
    ts_tq = ts_t.quantile_timeseries(q)
    ts_q = scalerP.inverse_transform(ts_tq)
    s = TimeSeries.to_series(ts_q)
    header = "Q" + format(int(q * 100), "02d")
    dfY_tft[header] = s
    if q == 0.5:
        ts_q50 = ts_q
        q50_RMSE = rmse(ts_q50, ts_val)
        q50_MAPE = mape(ts_q50, ts_val)
        print("RMSE:", f'{q50_RMSE:.2f}')
        print("MAPE:", f'{q50_MAPE:.2f}')


# call helper function predQ, once for every quantile
_ = [predQ(tft_predictions, q) for q in QUANTILES]

# move Q50 column to the left of the Actual column
col = dfY_tft.pop("Q50")
dfY_tft.insert(1, col.name, col)

dfY_tft.to_csv('result_TFT_1h_val.csv')
import plotly.graph_objs as go

def plot_actual_vs_q50(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Q50'], mode='lines', name='Q50'))

    fig.update_layout(title='Actual vs Q50 Load Data (Validation set)', xaxis_title='DateTime', yaxis_title='Load Value')

    fig.show()


# Plot the actual vs Q50 values
plot_actual_vs_q50(dfY_tft)

# Make predictions on the unseen test data
best_tft_test = best_model.predict(
    series=ts_ttrain.astype(np.float32).concatenate(ts_tval.astype(np.float32)),
    future_covariates=cov_t.astype(np.float32),
    n= len(ts_ttest),
    num_samples=num_eval_samples,
    n_jobs=3,
    verbose=True
)


tft_mape = mape(scalerP.inverse_transform(best_tft_test), ts_test)
tft_rmse = rmse(scalerP.inverse_transform(best_tft_test), ts_test)
tft_mse = mse(scalerP.inverse_transform(best_tft_test), ts_test)
tft_r2 = r2_score(scalerP.inverse_transform(best_tft_test), ts_test)
tft_smape = smape(scalerP.inverse_transform(best_tft_test), ts_test)
tft_mae = mae(scalerP.inverse_transform(best_tft_test), ts_test)

q50_RMSE = np.inf
q50_MAPE = np.inf
ts_q50 = None
pd.options.display.float_format = '{:,.2f}'.format
dfY_tft1 = pd.DataFrame()
dfY_tft1["Actual"] = TimeSeries.to_series(ts_test)


def predQ(ts_t, q):
    ts_tq = ts_t.quantile_timeseries(q)
    ts_q = scalerP.inverse_transform(ts_tq)
    s = TimeSeries.to_series(ts_q)
    header = "Q" + format(int(q * 100), "02d")
    dfY_tft1[header] = s
    if q == 0.5:
        ts_q50 = ts_q
        q50_RMSE = rmse(ts_q50, ts_test)
        q50_MAPE = mape(ts_q50, ts_test)
        print("RMSE:", f'{q50_RMSE:.2f}')
        print("MAPE:", f'{q50_MAPE:.2f}')


# call helper function predQ, once for every quantile
_ = [predQ(best_tft_test, q) for q in QUANTILES]

# move Q50 column to the left of the Actual column
col = dfY_tft1.pop("Q50")
dfY_tft1.insert(1, col.name, col)

import plotly.graph_objs as go

def plot_actual_vs_q50(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Q50'], mode='lines', name='Q50'))

    fig.update_layout(title='Actual vs Q50 Load Data (Test Set)', xaxis_title='DateTime', yaxis_title='Load Value')

    fig.show()


# Plot the actual vs Q50 values
plot_actual_vs_q50(dfY_tft1)


dfY_tft1.to_csv('result_TFT_1h.csv')

import matplotlib.pyplot as plt

# Prepare the data
n_points = len(dfY_tft1)
actual_data = dfY_tft1["Actual"][-n_points:]
q50_data = dfY_tft1["Q50"][-n_points:]

# Plot the actual data and Q50 predictions
plt.figure(figsize=(14, 6))
plt.plot(actual_data, label="Actual", color="red")
plt.plot(q50_data, label="Q50", color="blue")

# Plot the confidence intervals (quantiles) as shaded areas
for i, (q1, q2) in enumerate([(0.01, 0.95), (0.1, 0.9)]):
    col1 = f"Q{int(q1 * 100):02d}"
    col2 = f"Q{int(q2 * 100):02d}"
    plt.fill_between(
        dfY_tft1.index[-n_points:],
        dfY_tft1[col1][-n_points:],
        dfY_tft1[col2][-n_points:],
        color="blue",
        alpha=0.1 * (i + 1),
        label=f"{int(q1 * 100)}-{int(q2 * 100)}% CI",
    )

# Customize the plot
plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("Probabilistic Forecast")
plt.legend()
plt.show()



