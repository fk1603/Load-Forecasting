import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from darts.metrics import mape, rmse, mse, mae, r2_score, smape
from optuna.samplers import TPESampler
import holidays
import plotly.graph_objects as go
import logging
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit
import psutil
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure PyTorch for Tensor Cores and performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_dtype(torch.float32)  # Set default dtype to float32
    torch.use_deterministic_algorithms(True, warn_only=True)  # Enable deterministic mode with warnings
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info("Tensor Cores enabled with medium precision")
else:
    torch.set_default_dtype(torch.float32)  # Set default dtype to float32
    torch.use_deterministic_algorithms(True, warn_only=True)  # Enable deterministic mode with warnings
    logger.warning("CUDA is not available. Using CPU instead.")

# Suppress warnings
warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.precision", 2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.mode.chained_assignment = None

@jit(nopython=True)
def calculate_lags(data, lag_periods):
    """Optimized lag calculation using Numba."""
    result = np.zeros((len(data), len(lag_periods)))
    for i, lag in enumerate(lag_periods):
        result[lag:, i] = data[:-lag]
    return result

class LoadForecaster:
    def __init__(self, data_paths, model_dir="models"):
        """Initialize the LoadForecaster with paths to data files."""
        self.data_paths = data_paths
        self.data = None
        self.model = None
        self.scaler = None
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.n_workers = max(1, psutil.cpu_count(logical=False) - 1)
        self.quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    def load_and_preprocess_data(self):
        """Load and preprocess the load data."""
        try:
            logger.info("Loading load data...")
            
            # Load data
            self.data = pd.read_csv("processed_loadUTC_data.csv", header=0, parse_dates=["Time (UTC)"])
            self.data = self.data.rename(columns={"Time (UTC)": "Timestamp"})
            self.data.columns = ["Timestamp", "Forecast", "Actual_Load"]
            self.data = self.data.drop(columns=["Forecast"])
            
            # Load weather data
            dfw0 = pd.read_csv("weighted_avg_humidity.csv", header=0, parse_dates=["Timestamp"])
            dfw0.columns = ["Timestamp", "Humidity"]
            dfw1 = pd.read_csv("weighted_avg_solar.csv", header=0, parse_dates=["Timestamp"])
            dfw1.columns = ["Timestamp", "Solar"]
            dfw2 = pd.read_csv("weighted_avg_temp.csv", header=0, parse_dates=["Timestamp"])
            dfw2.columns = ["Timestamp", "Temperature"]
            
            # Merge weather data
            dfw0 = dfw0.merge(dfw1, on="Timestamp").merge(dfw2, on="Timestamp")
            
            # Convert timestamp format
            self.data["Timestamp"] = self.data["Timestamp"].str.extract(r"(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})")[0]
            self.data["Timestamp"] = pd.to_datetime(self.data["Timestamp"], format="%d.%m.%Y %H:%M")
            
            # Convert Actual_Load to float
            self.data["Actual_Load"] = self.data["Actual_Load"].astype(str).str.replace(",", "").astype(float)
            
            # Merge data
            self.data = self.data.merge(dfw0, on="Timestamp")
            
            # Remove old data
            self.data = self.data[~self.data["Timestamp"].dt.year.between(2015, 2020)]
            
            # Convert all numeric columns to float32
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                self.data[col] = self.data[col].astype('float32')
            
            logger.info("Data loading and preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {str(e)}")
            raise

    def add_features(self, lag_periods=[1, 2, 3, 24]):
        """Add time-based features and lagged features with optimized processing."""
        try:
            logger.info("Adding features...")
            
            self.data = self.data.sort_values('Timestamp')
            
            # Calculate lags using Numba-optimized function
            load_values = self.data["Actual_Load"].values
            lag_values = calculate_lags(load_values, lag_periods)
            
            # Add lagged features
            for i, lag in enumerate(lag_periods):
                self.data[f"Actual_Load_lag{lag}"] = lag_values[:, i]

            # Add time-based features
            self._add_time_features()
            
            # Drop NaN values and convert to float32
            self.data = self.data.dropna().reset_index(drop=True)
            float_cols = self.data.select_dtypes(include=['float64']).columns
            self.data[float_cols] = self.data[float_cols].astype('float32')

        except Exception as e:
            logger.error(f"Error adding features: {str(e)}")
            raise

    def _add_time_features(self):
        """Add time-based features to the dataset."""
        swe_holidays = holidays.Sweden(years=range(2021, 2025))
        self.data['Holiday'] = self.data['Timestamp'].dt.date.apply(
            lambda x: 1 if x in swe_holidays else 0
        )
        self.data['Hour'] = self.data['Timestamp'].dt.hour
        self.data['DayOfWeek'] = self.data['Timestamp'].dt.dayofweek
        self.data['Month'] = self.data['Timestamp'].dt.month
        self.data['Year'] = self.data['Timestamp'].dt.year

    def prepare_time_series(self):
        """Prepare time series data for modeling with year-based splits."""
        try:
            logger.info("Preparing time series data...")
            
            # Sort data by timestamp
            self.data = self.data.sort_values('Timestamp')
            
            # Create year-based splits
            train_data = self.data[self.data["Timestamp"].dt.year.between(2021, 2022)]
            val_data = self.data[self.data["Timestamp"].dt.year == 2023]
            test_data = self.data[self.data["Timestamp"].dt.year == 2024]
            
            # Log data ranges
            logger.info(f"Training data range: {train_data['Timestamp'].min()} to {train_data['Timestamp'].max()}")
            logger.info(f"Validation data range: {val_data['Timestamp'].min()} to {val_data['Timestamp'].max()}")
            logger.info(f"Test data range: {test_data['Timestamp'].min()} to {test_data['Timestamp'].max()}")
            
            # Initialize scaler with parallel processing
            self.scaler = Scaler(n_jobs=-1)
            
            # Create time series for target variable
            self.ts_train = TimeSeries.from_dataframe(
                train_data,
                time_col='Timestamp',
                value_cols=['Actual_Load']
            )
            self.ts_val = TimeSeries.from_dataframe(
                val_data,
                time_col='Timestamp',
                value_cols=['Actual_Load']
            )
            self.ts_test = TimeSeries.from_dataframe(
                test_data,
                time_col='Timestamp',
                value_cols=['Actual_Load']
            )
            
            # Scale the target variable
            self.ts_train = self.scaler.fit_transform(self.ts_train)
            self.ts_val = self.scaler.transform(self.ts_val)
            self.ts_test = self.scaler.transform(self.ts_test)
            
            # Convert all target series to float32
            self.ts_train = self.ts_train.astype('float32')
            self.ts_val = self.ts_val.astype('float32')
            self.ts_test = self.ts_test.astype('float32')
            
            # Create covariate series
            covariate_cols = ['Humidity', 'Solar', 'Temperature', 'Holiday',
                            'Actual_Load_lag1', 'Actual_Load_lag2', 'Actual_Load_lag3', 'Actual_Load_lag24']
            
            # Create all covariate series at once
            self.cov_train = TimeSeries.from_dataframe(
                train_data,
                time_col='Timestamp',
                value_cols=covariate_cols
            ).astype('float32')
            
            self.cov_val = TimeSeries.from_dataframe(
                val_data,
                time_col='Timestamp',
                value_cols=covariate_cols
            ).astype('float32')
            
            self.cov_test = TimeSeries.from_dataframe(
                test_data,
                time_col='Timestamp',
                value_cols=covariate_cols
            ).astype('float32')
            
            # Create future covariate series (time features)
            future_cov_cols = ['Hour', 'DayOfWeek', 'Month']
            
            # Create all future covariate series at once
            self.future_cov_train = TimeSeries.from_dataframe(
                train_data,
                time_col='Timestamp',
                value_cols=future_cov_cols
            ).astype('float32')
            
            self.future_cov_val = TimeSeries.from_dataframe(
                val_data,
                time_col='Timestamp',
                value_cols=future_cov_cols
            ).astype('float32')
            
            self.future_cov_test = TimeSeries.from_dataframe(
                test_data,
                time_col='Timestamp',
                value_cols=future_cov_cols
            ).astype('float32')
            
            logger.info("Time series preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Error preparing time series: {str(e)}")
            raise

    def build_model(self, params):
        """Build and train the TFT model with given parameters."""
        try:
            logger.info("Building TFT model...")
            
            # Configure early stopping exactly as in Ali.py
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0.005,
                patience=3,
                verbose=True
            )
            
            # Initialize model with exact same configuration as Ali.py
            self.model = TFTModel(
                input_chunk_length=params.get("input_chunk_length", 24),
                output_chunk_length=params.get("output_chunk_length", 24),
                hidden_size=params.get("hidden_size", 32),
                lstm_layers=params.get("lstm_layers", 2),
                num_attention_heads=params.get("num_attention_heads", 4),
                dropout=params.get("dropout", 0.1),
                optimizer_kwargs={"lr": params.get("lr", 1e-4)},
                batch_size=256,  # Fixed as in Ali.py
                n_epochs=70,     # Fixed as in Ali.py
                nr_epochs_val_period=1,  # Fixed as in Ali.py
                add_relative_index=True,
                add_encoders={
                    "cyclic": {
                        "past": ['hour', 'dayofweek', 'month'],
                        "future": ['hour', 'dayofweek', 'month']
                    },
                    "position": {
                        "past": ['relative'],
                        "future": ['relative']
                    }
                },
                likelihood=QuantileRegression(quantiles=self.quantiles),
                model_name="tft_model_1h",  # Same as Ali.py
                log_tensorboard=True,  # Same as Ali.py
                force_reset=True,
                save_checkpoints=True,  # Same as Ali.py
                pl_trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "callbacks": [early_stopping],
                    "precision": "32-true"
                }
            )
            
            # Train model with both past and future covariates
            logger.info("Training model...")
            self.model.fit(
                series=self.ts_train.astype('float32'),
                past_covariates=self.cov_train.astype('float32'),
                future_covariates=self.future_cov_train.astype('float32'),
                val_series=self.ts_val.astype('float32'),
                val_past_covariates=self.cov_val.astype('float32'),
                val_future_covariates=self.future_cov_val.astype('float32'),
                verbose=True
            )
            
            # Save model
            model_path = self.model_dir / "tft_model.pth"
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def optimize_hyperparameters(self, n_trials=50):
        """Optimize model hyperparameters using Optuna."""
        try:
            logger.info("Starting hyperparameter optimization...")

            def objective(trial):
                params = {
                    "input_chunk_length": trial.suggest_categorical("input_chunk_length", [12, 24, 48]),
                    "output_chunk_length": 24,  # Fixed as in Ali.py
                    "hidden_size": trial.suggest_categorical("hidden_size", [12, 24, 32]),
                    "lstm_layers": trial.suggest_int("lstm_layers", 1, 4),
                    "num_attention_heads": trial.suggest_int("num_attention_heads", 2, 6),
                    "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                }

                # Configure early stopping exactly as in Ali.py
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.005,
                    patience=3,
                    verbose=True
                )

                # Initialize model with float32 precision in trainer
                model = TFTModel(
                    input_chunk_length=params["input_chunk_length"],
                    output_chunk_length=params["output_chunk_length"],
                    hidden_size=params["hidden_size"],
                    lstm_layers=params["lstm_layers"],
                    num_attention_heads=params["num_attention_heads"],
                    dropout=params["dropout"],
                    optimizer_kwargs={"lr": params["lr"]},
                    batch_size=256,  # Fixed batch size as in Ali.py
                    n_epochs=70,     # Fixed epochs as in Ali.py
                    nr_epochs_val_period=1,  # Validation period as in Ali.py
                    add_relative_index=True,
                    add_encoders={
                        "cyclic": {
                            "past": ['hour', 'dayofweek', 'month'],
                            "future": ['hour', 'dayofweek', 'month']
                        },
                        "position": {
                            "past": ['relative'],
                            "future": ['relative']
                        }
                    },
                    likelihood=QuantileRegression(quantiles=self.quantiles),
                    model_name="tft_model_1h",
                    log_tensorboard=True,
                    force_reset=True,
                    save_checkpoints=True,
                    pl_trainer_kwargs={
                        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                        "callbacks": [early_stopping],
                        "precision": "32-true"
                    }
                )

                # Train the model
                model.fit(
                    series=self.ts_train.astype('float32'),
                    past_covariates=self.cov_train.astype('float32'),
                    future_covariates=self.future_cov_train.astype('float32'),
                    val_series=self.ts_val.astype('float32'),
                    val_past_covariates=self.cov_val.astype('float32'),
                    val_future_covariates=self.future_cov_val.astype('float32'),
                    verbose=True
                )

                # Make predictions in chunks of output_chunk_length
                predictions = []
                total_chunks = (len(self.ts_val) + params["output_chunk_length"] - 1) // params["output_chunk_length"]
                
                logger.info(f"Making predictions in {total_chunks} chunks...")
                
                for i in range(0, len(self.ts_val), params["output_chunk_length"]):
                    # Get the current chunk of validation data
                    chunk_size = min(params["output_chunk_length"], len(self.ts_val) - i)
                    logger.info(f"Processing chunk {i//params['output_chunk_length'] + 1}/{total_chunks} (size: {chunk_size})")
                    
                    # Calculate the start time for future covariates
                    # We need to start from the end of training data
                    chunk_start_time = self.ts_train.time_index[-1] - pd.Timedelta(hours=params["input_chunk_length"])
                    chunk_end_time = self.ts_val.time_index[i + chunk_size - 1]
                    
                    logger.info(f"Future covariates range: {chunk_start_time} to {chunk_end_time}")
                    
                    # Create time-based features for the prediction period including overlap
                    chunk_future_cov = TimeSeries.from_times_and_values(
                        times=pd.date_range(start=chunk_start_time, end=chunk_end_time, freq='H'),
                        values=np.column_stack([
                            pd.date_range(start=chunk_start_time, end=chunk_end_time, freq='H').hour,
                            pd.date_range(start=chunk_start_time, end=chunk_end_time, freq='H').dayofweek,
                            pd.date_range(start=chunk_start_time, end=chunk_end_time, freq='H').month
                        ])
                    ).astype('float32')
                    
                    # Make prediction for this chunk
                    chunk_predictions = model.predict(
                        n=chunk_size,
                        series=self.ts_train,
                        past_covariates=self.cov_train,
                        future_covariates=chunk_future_cov
                    )
                    predictions.append(chunk_predictions)
                
                # Combine predictions using the + operator
                final_predictions = predictions[0]
                for pred in predictions[1:]:
                    final_predictions = final_predictions + pred

                return mape(self.ts_val, final_predictions)

            def print_callback(study, trial):
                logger.info(f"Current value: {trial.value}, Current params: {trial.params}")
                logger.info(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

            # Create study with persistent storage
            study = optuna.create_study(
                study_name="T15",
                storage="sqlite:///optimization.db",
                load_if_exists=True,
                sampler=TPESampler(seed=42),
                direction="minimize"
            )
            
            # Run optimization with callback
            study.optimize(objective, n_trials=n_trials, callbacks=[print_callback])

            logger.info(f"Best trial: {study.best_trial.value}")
            logger.info(f"Best parameters: {study.best_params}")

            return study.best_params

        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise

    def make_predictions(self, n_samples=500):
        """Make predictions for the test period."""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            logger.info("Making predictions...")
            
            # Ensure we have enough historical data
            if len(self.ts_train) < self.model.input_chunk_length:
                raise ValueError("Insufficient historical data for predictions")
            
            # Make predictions with both past and future covariates
            predictions = self.model.predict(
                n=len(self.ts_test),
                series=self.ts_train,
                past_covariates=self.cov_train,
                future_covariates=self.future_cov_train,
                num_samples=n_samples
            )
            
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions)
            
            logger.info(f"Predictions shape: {predictions.shape()}")
            logger.info(f"Predictions time range: {predictions.time_index[0]} to {predictions.time_index[-1]}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate_predictions(self, predictions):
        """Evaluate model predictions using various metrics."""
        try:
            logger.info("Evaluating predictions...")
            
            # Ensure predictions and test data are aligned
            test_data = self.scaler.inverse_transform(self.ts_test)
            predictions = predictions.slice_intersect(test_data)
            
            # Calculate metrics
            metrics = {
                "MAPE": mape(test_data, predictions),
                "RMSE": rmse(test_data, predictions),
                "MSE": mse(test_data, predictions),
                "MAE": mae(test_data, predictions),
                "R2": r2_score(test_data, predictions),
                "SMAPE": smape(test_data, predictions)
            }
            
            # Log metrics
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            raise

    def plot_predictions(self, predictions, n_points=None):
        """Plot actual vs predicted values."""
        try:
            logger.info("Creating prediction plot...")
            
            # Get actual values
            actual = self.scaler.inverse_transform(self.ts_test)
            
            # Create plot
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=actual.time_index,
                y=actual.values(),
                name="Actual",
                line=dict(color="blue")
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=predictions.time_index,
                y=predictions.values(),
                name="Predicted",
                line=dict(color="red")
            ))
            
            # Update layout
            fig.update_layout(
                title="Load Forecast: Actual vs Predicted",
                xaxis_title="Time",
                yaxis_title="Load (MW)",
                height=800,
                width=1200,
                showlegend=True,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            # Save plot
            plot_path = self.model_dir / "predictions.html"
            fig.write_html(str(plot_path))
            logger.info(f"Plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            raise

    def load_model(self):
        """Load a trained model from disk."""
        try:
            model_path = self.model_dir / "best_model.pt"  # Changed to match existing file name
            if not model_path.exists():
                logger.warning("No saved model found")
                return False
            
            logger.info("Loading model...")
            self.model = TFTModel.load(model_path)
            
            # Verify model can make predictions
            test_pred = self.model.predict(
                n=1,
                series=self.ts_train,
                past_covariates=self.cov_train
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

def main():
    """Main execution function."""
    try:
        # Define data paths
        data_paths = {
            'load_data': "processed_loadUTC_data.csv",
            'humidity_data': "weighted_avg_humidity.csv",
            'solar_data': "weighted_avg_solar.csv",
            'temperature_data': "weighted_avg_temp.csv"
        }
        
        # Initialize forecaster
        forecaster = LoadForecaster(data_paths)
        
        # Process data
        forecaster.load_and_preprocess_data()
        forecaster.add_features()
        forecaster.prepare_time_series()
        
        # Model selection
        while True:
            choice = input("\nDo you want to:\n1. Load existing model\n2. Train new model\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                if forecaster.load_model():
                    break
                print("No saved model found. Please choose option 2 to train a new model.")
            elif choice == "2":
                print("Training new model...")
                best_params = forecaster.optimize_hyperparameters(n_trials=50)
                forecaster.build_model(best_params)
                break
        
        # Make and evaluate predictions
        predictions = forecaster.make_predictions()
        metrics = forecaster.evaluate_predictions(predictions)
        forecaster.plot_predictions(predictions)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()  