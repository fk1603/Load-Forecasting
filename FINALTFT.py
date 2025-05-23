# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Temporal Fusion Transformer (TFT)
#
# This notebook covers the impementation of a TFT model for load forecasting purposes.
#
# The model is implemented using the Darts python package
#
# Please pip install the requirements.txt file before running
#
# Authors: Ali Kamran, (Andreas Liiv, Florian Kühn)

# %% id="4a7312ed"
# Import required modules
from darts.explainability.tft_explainer import TFTExplainer
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from optuna.exceptions import TrialPruned
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings  # Suppress warnings for cleaner output
import os  # Operating system interface
import optuna  # Hyperparameter optimization
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation
import torch  # PyTorch for deep learning
from darts.utils.likelihood_models.base import LikelihoodType  # Darts likelihood types
from darts.metrics import mape, rmse, mse, mae, r2_score, smape  # Darts metrics
from darts import TimeSeries  # Darts TimeSeries object
from darts.dataprocessing.transformers import Scaler  # Data scaling
from darts.metrics import mape  # Mean Absolute Percentage Error
import holidays  # Holiday calculations
import torchmetrics  # PyTorch metrics
from torchmetrics.collections import MetricCollection  # Metric collection
from darts.models import TFTModel  # Temporal Fusion Transformer model
# Quantile regression for probabilistic forecasting
from darts.utils.likelihood_models import QuantileRegression
import torch.optim as optim  # PyTorch optimizers

# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium')
device0 = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")  # Set device
# warnings.filterwarnings("ignore")  # Ignore warnings
print(torch.cuda.is_available())


# %% id="ec7adecd"
def read_load_data():
    '''
    Reads load data from file 'processed_loadUTC_data.csv'.
    Renames columns for better handling in future code.

    Returns: DataFrame containing load data
    '''

    base_dir = os.getcwd()
    load_file_path = os.path.join(base_dir, 'processed_loadUTC_data.csv')
    load_data = pd.read_csv(load_file_path)
    load_data.rename(columns={
        "Actual Total Load [MW] - BZN|SE4": "Actual Load",
        "Time (UTC)": "Time"
    }, inplace=True)
    load_data['Time'] = pd.to_datetime(
        load_data['Time'].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')
    load_data.set_index('Time', inplace=True)
    load_data.drop(
        'Day-ahead Total Load Forecast [MW] - BZN|SE4', axis=1, inplace=True)
    return load_data


data = read_load_data()


# %% id="8ee8a9ec"
def import_weather_data(df, solar_power=False):
    '''
    The weighted averages of weather data (incl. temperature, humidity and solar irradiation) is read and added to the main dataframe.
    If "solar_power" is given as 'True', solar irradiation data is replaced by an estimate of total solar power generation.

    Parameters:
        df: pandas DataFrame
        solar_power: boolean for converting solar irradiation to solar power if 'True'

    Returns: input DataFrame appended with weather data
    '''

    base_dir = os.getcwd()
    temperature_file_path = os.path.join(base_dir, 'weighted_avg_temp.csv')
    humidity_file_path = os.path.join(base_dir, 'weighted_avg_humidity.csv')
    solar_file_path = os.path.join(base_dir, 'weighted_avg_solar.csv')
    avg_temperature_data = pd.read_csv(temperature_file_path)
    avg_humidity_data = pd.read_csv(humidity_file_path)
    avg_solar_data = pd.read_csv(solar_file_path)
    df['Temperature'] = avg_temperature_data['weighted_avg'].values
    df['Humidity'] = avg_humidity_data['weighted_avg'].values
    df['Solar_Irrad'] = avg_solar_data['weighted_avg'].values

    if solar_power:
        tot_installed_capacity = np.array([
            2.52, 5.04, 7.56, 10.08, 12.6, 18.6, 29.2, 41.53, 69.55, 85
        ]) * 1e6
        installed_capacity_house = 10 * 1e3
        n_installations = tot_installed_capacity / installed_capacity_house
        panel_per_house = 24
        area_per_panel = 2

        years = np.arange(2015, 2025)
        total_area_panels = n_installations * panel_per_house * area_per_panel
        area_by_year = dict(zip(years, total_area_panels))

        panel_efficiency = 0.2
        df['year'] = df.index.year
        df['Solar'] = df['Solar_Irrad'] * \
            df['year'].map(area_by_year) * panel_efficiency
        df.drop(columns='year', inplace=True)
    else:
        df['Solar'] = df['Solar_Irrad']
    df.drop(columns='Solar_Irrad', inplace=True)
    return df


data = import_weather_data(data, solar_power=True)


# %% id="49ab2b66"
def import_swedish_holidays(df):
    '''
    Imports Swedish holidays into column 'is_holiday' of input DataFrame

    Parameters:
        df: pandas Dataframe

    Returns: DataFrame appended with Swedish holidays
    '''
    years = np.arange(2015, 2025)
    raw_holidays = []
    for date, name in sorted(holidays.Sweden(years=years).items()):
        if name != "Söndag":
            raw_holidays.append(date)
        raw_holidays.append(date)
    holiday_dates = set(raw_holidays)
    df['is_holiday'] = pd.Series(df.index.date).isin(
        holiday_dates).astype(int).values
    return df


data = import_swedish_holidays(data)


# %% id="4ec1d3df"
def create_date_features(df):
    '''
    Creates date features marking the hour of the day

    Parameters:
        df: pandas DataFrame

    Returns: DataFrame appended with hour
    '''
    df['hour'] = df.index.hour
    return df


data = create_date_features(data)


# %% id="db4b08a6"
def add_noise_to_covariates(df, noise_level=0.05):
    """
    Adds Gaussian noise to weather-related covariates ('Temperature', 'Humidity', 'Solar').

    The function adds random noise with a standard deviation proportional to the value
    (for 'Solar') or to the column's standard deviation (for 'Temperature' and 'Humidity').
    This is useful for simulating measurement uncertainty or for data augmentation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing weather covariates.
    noise_level : float, optional
        Standard deviation of the Gaussian noise as a fraction of the value (default is 0.05).

    Returns
    -------
    pandas.DataFrame
        DataFrame with noisy weather covariates.
    """
    np.random.seed(42)  # Set random seed for reproducibility
    stochastic_cols = ["Temperature", "Humidity", "Solar"]
    df_noisy = df.copy()
    for col in stochastic_cols:
        if col in df_noisy.columns:
            if col == "Solar":
                base = df_noisy[col].values
                noise = np.random.normal(0, noise_level * base)
                df_noisy[col] += noise
                df_noisy[col] = df_noisy[col].clip(lower=0)
            else:
                std = df_noisy[col].std()
                noise = np.random.normal(
                    0, noise_level * std, size=len(df_noisy))
                df_noisy[col] += noise
    return df_noisy


# Retain the original data without noise
clean_future_data = data.copy()
future_data = add_noise_to_covariates(data)


# %% colab={"base_uri": "https://localhost:8080/"} id="d0887cd0" outputId="92f62171-9534-4406-9b48-95eff33c0ae7"
def create_lag_features(df, load="Actual Load", nan=False):
    '''
    Creates lagged covariate and load features. Can fill target value (load) lags with NaNs if nan is 'True'.

    df: pandas DataFrame
    nan: fills all load lags with NaNs if 'True'
    '''

    df = df.copy()
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_temp_{lag}'] = df["Temperature"].shift(lag)
        df[f'lag_humid_{lag}'] = df["Humidity"].shift(lag)
        df[f'lag_solar_{lag}'] = df["Solar"].shift(lag)
        if lag <= 3:
            df[f'lag_hour_{lag}'] = df["hour"].shift(lag)
        if nan:
            df[f'lag_{lag}'] = np.nan
        else:
            df[f'lag_{lag}'] = df[load].shift(lag)
    for day in range(2, 8):
        lag = day * 24
        if nan:
            df[f'lag_{lag}'] = np.nan
        else:
            df[f'lag_{lag}'] = df[load].shift(lag)
    return df


data = create_lag_features(data)
data = data.dropna()  # first week in 2015 dropped to enable lag features
print(data.columns.tolist())
# droping the lagged features not used in the model
data.drop(
    columns=[
        'lag_1',
        'lag_2',
        'lag_3',
        'lag_6',
        'lag_12',
        'lag_hour_1',
        'lag_hour_2',
        'lag_hour_1',
        'lag_hour_3',
    ],
    inplace=True)
# future_data = create_lag_features(future_data, nan=True)
future_data = future_data[future_data.index >= data.index.min()]

# %% colab={"base_uri": "https://localhost:8080/"} id="f51550a5" outputId="511ac1c9-dbcd-4e57-e3d3-5fa82285c10f"
# drop all the data before 2022
data = data[data.index >= "2022-01-01"]

future_data = future_data[future_data.index >= "2022-01-01"]
clean_future_data = clean_future_data[clean_future_data.index >= "2022-01-01"]
# print(future_data.index)  # Should be DatetimeIndex

series = TimeSeries.from_dataframe(data, value_cols="Actual Load")
# print(series.columns.tolist())   # Lists all column names

# %% id="ae7cd138"
# Split dates for training and validation sets
training_cutoff = pd.Timestamp("2023-12-31 23:00:00")
train_cov_cutoff = pd.Timestamp("2024-01-01 00:00:00")
validating_cutoff = pd.Timestamp("2024-06-30 23:00:00")

# %% colab={"base_uri": "https://localhost:8080/"} id="9f111b0c" outputId="e2596bf5-1299-4497-acec-e14ca302aaef"
# Split after training_cutoff
train, rest = series.split_after(training_cutoff)

# Split the remaining data after validating_cutoff
val, test = rest.split_after(validating_cutoff)
print('Train Series:', train.start_time(), train.end_time())
print('Val Series:', val.start_time(), val.end_time())
print('Test Series:', test.start_time(), test.end_time())

# Normalize the time series (scaling)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)
test_transformed = transformer.transform(test)

# %% colab={"base_uri": "https://localhost:8080/"} id="9e8c3e6f" outputId="1a432225-7bac-4b39-df53-0d5bd87297ea"
# Select all columns except 'Actual Load'
value_cols = [col for col in data.columns if col != "Actual Load"]

# Create the TimeSeries (assuming index is datetime or time_col is defined)
past_covariates = TimeSeries.from_dataframe(data, value_cols=value_cols)
# Normalize historical covariates
scaler_covs = Scaler()
scaler_covs.fit(past_covariates[:training_cutoff])
past_covariates_transformed = scaler_covs.transform(past_covariates)
# Use same cutoff points for consistency
train_past_covs, rest_past_covs = past_covariates_transformed.split_after(
    training_cutoff)
val_past_covs, test_past_covs = rest_past_covs.split_after(validating_cutoff)
last24_ = train_past_covs[-24:]
val_past_covs = last24_.append(val_past_covs)

# %% colab={"base_uri": "https://localhost:8080/"} id="3fe1827c" outputId="34fb65e4-b923-4480-96e6-0612f7051ad4"
# Apply the same approach for future covariates:
# Right now we are using the same data as past covariates

# Drop "Actual Load" from value columns
value_cols = [col for col in future_data.columns if col != "Actual Load"]

# Create the TimeSeries
future_covariates = TimeSeries.from_dataframe(
    future_data, value_cols=value_cols)
clean_future_cov = TimeSeries.from_dataframe(
    clean_future_data, value_cols=value_cols)
# Clean future covariates for training as its not intended the model to
# learn the noise pattern otherwise it performs better
clean_train_future_covs, rest_future_covs = clean_future_cov.split_after(
    training_cutoff)
noisy_data, rest_data = future_covariates.split_after(training_cutoff)
print(clean_train_future_covs.end_time())
print(rest_data.start_time())
future_covariates = clean_train_future_covs.append(rest_data)

# Apply the same approach for future covariates:
scaler_covs_future = Scaler()
# Fit on training data
scaler_covs_future.fit(future_covariates[:training_cutoff])

# Transform all future covariates
future_covariates_transformed = scaler_covs_future.transform(future_covariates)

# Split transformed future covariates into training, validation, and test sets
train_future_covs, rest_future_covs = future_covariates_transformed.split_after(
    training_cutoff)
val_future_covs, test_future_covs = rest_future_covs.split_after(
    validating_cutoff)

last24_ = train_future_covs[-24:]

val_future_covs = last24_.append(val_future_covs)
print(val_future_covs.start_time(), val_future_covs.end_time())

# add the first 24 hours of val future covariates to the end of the train
# future covariates
next48 = val_future_covs[:48]
next24 = next48[-24:]
train_future_covs = train_future_covs.append(next24)
print(train_future_covs.start_time(), train_future_covs.end_time())

# %% colab={"base_uri": "https://localhost:8080/"} id="7e8667fd" outputId="b4dfa723-fe03-4e66-caa8-80b25e63027b"
# Define quantiles for QuantileRegression
quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
             0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,]

# %% id="cb3ef6c5"


def encode_year(index):
    return pd.Series(index.year, index=index)

# Define callback function


# stop training when validation loss does not decrease more than 0.05
# (`min_delta`) over


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
    optuna_callback = CustomPyTorchLightningPruningCallback(
        trial, monitor="val_loss")
    early_stopper = EarlyStopping(
        monitor="val_loss",  # Using train_loss instead of val_loss might not be a good idea
        min_delta=0.005,
        patience=3,
        verbose=True
    )
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [12, 24, 32])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    input_chunk_length = trial.suggest_categorical(
        "input_chunk_length", [12, 24, 48])
    batch_size = 256

    model = TFTModel(
        model_name=f"TFT_trial_{trial.number}",
        input_chunk_length=input_chunk_length,
        output_chunk_length=24,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        batch_size=256,
        optimizer_kwargs={"lr": learning_rate},
        n_epochs=20,  # Keep it low for faster tuning
        likelihood=QuantileRegression(quantiles=quantiles),
        random_state=42,
        add_encoders={
            'cyclic': {'future': ['month', 'day', 'weekday', 'dayofweek', 'hour', 'dayofyear', 'weekofyear'],
                       'past': ['month', 'day', 'weekday', 'dayofweek', 'hour', 'dayofyear', 'weekofyear']},
            'datetime_attribute': {'future': ['hour', 'dayofweek']},
            'position': {'past': ['relative'], 'future': ['relative']},
            'custom': {'past': [encode_year]},
            'transformer': Scaler(),
            'tz': 'UTC'},

        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else None,
            "enable_progress_bar": True,
            "callbacks": [early_stopper, optuna_callback],
        },
        force_reset=True
    )

    model.fit(
        train_transformed[24:],
        past_covariates=train_past_covs,
        future_covariates=train_future_covs,
        val_series=val_transformed,
        val_past_covariates=val_past_covs,
        val_future_covariates=val_future_covs,
        verbose=True,
    )

    # Forecast and compute MAPE
    pred = model.predict(
        n=len(val_transformed),
        series=train_transformed,
        past_covariates=val_past_covs,
        future_covariates=val_future_covs,
        num_samples=100)
    score = mape(val_transformed, pred)
    return score  # Optuna will minimize this


# %% colab={"base_uri": "https://localhost:8080/"} id="801838fd" outputId="2d0240fe-11e0-4259-8d2d-9ade1e06584e"
print(train_transformed.end_time())
print(val_transformed.start_time())
print(val_past_covs.start_time())

# %% colab={"base_uri": "https://localhost:8080/"} id="44f5fbc1" outputId="fb20cbce-cb1a-4c4e-8898-654ab12d9bae"
forecast_horizon = 24


LOAD = False     # True for hyperparameter tuning
if LOAD:
    print('Hyperparameter tuning:')
    study = optuna.create_study(
        study_name="T22",
        storage="sqlite:///optimization04.db",
        load_if_exists=True,
        direction="minimize")
    study.optimize(objective, n_trials=10, )  # run for 20 trials or 1 hour
    print("Best MAPE:", study.best_value)
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params
    print(best_params)
else:
    print('loading the study:')
    study = optuna.create_study(
        study_name="T22",
        storage="sqlite:///optimization04.db",
        load_if_exists=True,
        direction="minimize")
    best_params = study.best_trial.params
    early_stopper = EarlyStopping(
        monitor="val_loss",  # Using train_loss instead of val_loss might not be a good idea
        min_delta=0.005,
        patience=3,
        verbose=True
    )
    # Define model parameters
    # Define model with single pl_trainer_kwargs
    my_model = TFTModel(

        model_name="TFT",
        input_chunk_length=best_params["input_chunk_length"],
        output_chunk_length=forecast_horizon,
        hidden_size=best_params["hidden_size"],
        lstm_layers=best_params["lstm_layers"],
        dropout=best_params["dropout"],
        add_encoders={
            'cyclic': {'future': ['month', 'day', 'weekday', 'dayofweek', 'hour', 'dayofyear', 'weekofyear'],
                       'past': ['month', 'day', 'weekday', 'dayofweek', 'hour', 'dayofyear', 'weekofyear']},
            'datetime_attribute': {'future': ['hour', 'dayofweek']},
            'position': {'past': ['relative'], 'future': ['relative']},
            'custom': {'past': [encode_year]},
            'transformer': Scaler(),
            'tz': 'UTC'},



        batch_size=256,
        optimizer_kwargs={"lr": best_params["lr"]},
        n_epochs=30,  # now you can train longer
        likelihood=QuantileRegression(quantiles=quantiles),
        random_state=42,
        save_checkpoints=True,
        add_relative_index=True,
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": [0] if torch.cuda.is_available() else None,
            "enable_progress_bar": True,
            "callbacks": [early_stopper],
        },
        force_reset=True
    )

# %% id="7cea92fe"
# del my_model

# %% colab={"base_uri": "https://localhost:8080/", "height": 721, "referenced_widgets": ["874e7b6780234affab69232179e6c87e", "cf8d115b65a84120b7f8612988c76ecb", "51b61cab394a4a4ba23cfa5582913a4d", "0a5665c2d91c4b16925e3619c682300e", "9d3e8f6f89824447bd1f1353eef22023", "a1bbd66c2aec4a6989062d035222ca82", "2b30fdaeeedf4bb4b78da6ea99c1b16c", "0995b72197bc449c8357118e2ffb986c", "6dcce142b9214e2aa21b1f777578e20f", "29025d7922c54cce8463c3e1c0a1a8e5", "acec1cba57904d85a8840c539a80adbb", "fd09ef5aecde4d34bbf650675b358c11", "c33356c7473b4e43bb34572483ba07ce", "7838597fa8a4410683a901f9b4b1d95e", "d509f9e0d9e34d1ca3053670cb7756de", "3968416e3a7b40da8df000d82f0afded", "f7669d8bf06d46cabdeb0b5ad2c68aff", "da60c9d6624e44438adfa932caf4d16a", "c76e7ab56aa74ab5ab3a11e15d2c90ee", "06c5471419ff492ca9b99089baa886c1", "9e7daa588b1c4f5396ec124b6e4c21c4", "a9b809c4a5ac48e4a0cb39c5d2b382df", "167344177c4346d0bd7ef41d7af9a1c0", "60516e3e035c4d03920350cad54846b9", "ec06b604af344079b93afa1b6f0f408e", "a168d360f3a748269275939ea80de6ac", "e7af1b4720454002b122ee01c4f34d33", "c8885e6bc50c4b3095aceb987588fd55", "0b68d5a00af94fb78596366fcd1326aa", "c4c476d172844619a02b40e0b571e4c9", "06211226b58c434baa7526616f8ffce1", "adc325e6551b45e6ae1d6cfc358886dd", "b8826918360f430caf73cdd8d7bb2147", "d05e325c55ff4fe2bd875aa7dc03c938", "7c74a37fa6064f9fa1eb559ac81fb8cd", "66a48cd3fa254bc396862d5e9d161826", "3f82898b26a94afcbf369ab6b6982215", "0928251a96a74ce99a8f90fb69441fad", "9be0a8350a324fb78d3d49a4b500e696", "7eb1c8eaa55d4d8eba69ab6e921878d8", "17840db8578d447d95dd4acd61eacbf9", "e8e762eb04e94d33be19fa76ce76702e", "f7c83f3003f848cebb28803c0f50ad77", "f7ead877c2f9440e80a28f58247f54cc", "0dba04b3a8214f68971f33ecc9af5814", "d5883734bef341649d373b0588ef6a34", "3c7403c7c0aa4f34b8ef2d1cd9d2f0d0", "6469abf8c648404681b4efb439d6ab1c", "a17b0f1d701e438fb0350f94d5867e19", "97c8cde3246042c9aa946c0b78974263", "8a4240c5167b45e6b2f2e40a85d32a8a", "8d1b293955c647ee8dc57e335adfb1ca", "31f8305661c14875ae73a7f7f0727547", "9f828b723fd7452e84613ec79dbf6ae0", "28940f8ca0fa4f33a6972df029ec48b4"]} id="987b3ff1" outputId="4278f1c3-8cc5-4ce8-e6fb-6e417fce5c20"
train_again = False  # Change this to False if you want to retrain

if train_again:
    my_model.fit(
        train_transformed[24:],
        past_covariates=train_past_covs,
        future_covariates=train_future_covs,
        val_series=val_transformed,
        val_past_covariates=val_past_covs,
        val_future_covariates=val_future_covs,
        verbose=True,
    )
    my_model.save("slutlig.pt")
    print("TFT model saved to slutlig.pt")
else:
    print("Skipping training; using existing model if available.")

my_model = None

# Load the model
try:
    # Add all necessary classes to safe globals
    torch.serialization.add_safe_globals([
        MetricCollection,
        QuantileRegression,
        LikelihoodType,
        optim.Adam,
        TFTModel,
        torchmetrics.Metric,
        torchmetrics.MetricCollection,
        getattr
    ])

    # Load from the correct path - use the same path as in save()
    my_model = TFTModel.load(
        "slutlig.pt",  # MUST match exactly with my_model.save("my_tft_model")
        map_location=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    raise

# %% colab={"base_uri": "https://localhost:8080/", "height": 477} id="0a20c51c" outputId="433ca71d-5802-41ed-b1a7-70c20929f2e0"


# For a pandas DataFrame or Series
print("Start time:", val.start_time())
print("End time:", val.end_time())
print("End time:", test_past_covs.end_time())
print("Start time:", test_future_covs.start_time())
print("End time:", test_future_covs.end_time())

# Select the first 24 time steps from test_past_covs
# Select the last 24 entries from val_past_covs
last_24_covs = val_past_covs[-24:]

# Concatenate the last 24 entries at the start of test_past_covs
test_past_covs = last_24_covs.append(test_past_covs)
print("Start time:", test_past_covs.start_time())

# Select the last 24 entries from val_past_covs
last_24_covs = val_future_covs[-24:]

# Concatenate the last 24 entries at the start of test_past_covs
test_future_covs = last_24_covs.append(test_future_covs)
# Get the first time step of test_transformed

# Append it to val_transformed
next_step = test_transformed[:24]

val_extended = val_transformed.append(next_step)

# %% [markdown]
# # Full Forecast

# %%
forecast_date = pd.Timestamp("2024-07-06 23:00:00")
the_series, discard = test_transformed.split_after(forecast_date)

# %%
forecast = my_model.predict(
    n=len(test) - 7 * 24,
    series=the_series,
    past_covariates=past_covariates_transformed,
    future_covariates=future_covariates_transformed,
    num_samples=100)

# %% colab={"base_uri": "https://localhost:8080/"} id="580e94b8" outputId="e4801465-6d13-4710-ba50-78c5da05f8ed"
# Inverse transform forecast and train data
forecast_inv = transformer.inverse_transform(forecast)
forecast_inv = forecast_inv
start, end = forecast_inv.start_time(), forecast_inv.end_time()
test_series = test.slice(start, end)
long_forecast = forecast_inv.copy()
long_forecast_actual = test_series.copy()
# Calculate evaluation metrics
tft_mape = mape(test_series, forecast_inv)  # MAPE
tft_rmse = rmse(test_series, forecast_inv)  # RMSE
tft_mae = mae(test_series, forecast_inv)  # MAE

# Print results
print(f"MAPE: {tft_mape:.2f}")
print(f"MAE: {tft_mae:.2f}")
print(f"RMSE: {tft_rmse:.2f}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 530} id="ec0bf2a6" outputId="34d4ddf4-f772-4854-ba49-00c52132aa60"
# plot actual series


plt.figure(figsize=(12, 6))
test_series[: test_series.end_time()].plot(label="actual")
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
# plot prediction with quantile ranges
forecast_inv.plot(
    low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
)
forecast_inv.plot(low_quantile=low_q, high_quantile=high_q,
                  label=label_q_inner)

plt.title(f"MAPE: {mape(test_series, forecast_inv):.2f}%")
plt.legend()

# %% [markdown]
# # Short Forecasts

# %%
# Specify the date for the forecast and number of days to forecast
forecast_dates = [
    pd.Timestamp("2024-07-21 23:00:00"),
    pd.Timestamp("2024-10-13 23:00:00"),
    pd.Timestamp("2024-10-27 23:00:00"),
    pd.Timestamp("2024-12-01 23:00:00"),
    pd.Timestamp("2024-12-22 23:00:00")
]
for i in range(len(forecast_dates)):
    forecast_date = forecast_dates[i]
    number_of_days = 7
    hours = number_of_days * 24
    # split the target series in the same way after forecast date
    the_series, discard = test_transformed.split_after(forecast_date)
    print(the_series.end_time())
    forecast = my_model.predict(
        n=hours,
        series=the_series,
        past_covariates=past_covariates_transformed,
        future_covariates=future_covariates_transformed,
        num_samples=100)
    # Inverse transform forecast and train data
    forecast_inv = transformer.inverse_transform(forecast)
    forecast_inv = forecast_inv
    start, end = forecast_inv.start_time(), forecast_inv.end_time()
    test_series = test.slice(start, end)

    # Calculate evaluation metrics
    tft_mape = mape(test_series, forecast_inv)  # MAPE
    tft_rmse = rmse(test_series, forecast_inv)  # RMSE
    tft_mse = mse(test_series, forecast_inv)  # MSE
    tft_r2 = r2_score(test_series, forecast_inv)  # R^2
    tft_smape = smape(test_series, forecast_inv)  # SMAPE
    tft_mae = mae(test_series, forecast_inv)  # MAE

    # Print results
    print(f"MAPE: {tft_mape:.2f}")
    print(f"MAE: {tft_mae:.2f}")
    print(f"RMSE: {tft_rmse:.2f}")

    # plot actual series

    plt.figure(figsize=(12, 6))
    test.slice(forecast_inv.start_time(),
               forecast_inv.end_time()).plot(label="actual")
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
    # plot prediction with quantile ranges
    forecast_inv.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    forecast_inv.plot(low_quantile=low_q,
                      high_quantile=high_q, label=label_q_inner)

    plt.title(f"MAPE: {mape(test_series, forecast_inv):.2f}%")
    plt.legend()

# %%

# Calculate daily MAPEs in 24-hour chunks
chunk_size = 24  # Number of hours per chunk
n_chunks = len(long_forecast) // chunk_size

daily_mapes = []

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size

    test_chunk = test_series[start_idx:end_idx]
    forecast_chunk = forecast_inv[start_idx:end_idx]

    error = mape(test_chunk, forecast_chunk)
    daily_mapes.append(error)

# Generate start timestamps for each 24-hour chunk
chunk_starts = [long_forecast.start_time() + pd.Timedelta(hours=chunk_size * i)
                for i in range(n_chunks)]

# Create a DataFrame to hold daily MAPE values with their corresponding
# start dates
df_mape = pd.DataFrame({
    "start_date": chunk_starts,
    "daily_mape": daily_mapes
})

# Plot the distribution of daily MAPEs using a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(y="daily_mape", data=df_mape, color="lightblue")
plt.title("Daily MAPE Distribution (24-hour chunks)")
plt.ylabel("MAPE (%)")
plt.grid(True)
plt.show()

# Save the daily MAPE values along with their start dates to a CSV file
df_mape.to_csv("daily_mape_TFT.csv", index=False)

# %%

# requires `background` if model was trained on multiple series
explainer = TFTExplainer(my_model)
results = explainer.explain()
# plot the results
explainer.plot_attention(results, plot_type="all")
explainer.plot_variable_selection(results)

# %%
# !jupytext --to py FINALTFT.ipynb && autopep8 --in-place --aggressive --aggressive FINALTFT.py && jupytext --to notebook FINALTFT.py
