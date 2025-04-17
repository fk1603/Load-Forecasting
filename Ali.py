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
import time
pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format
print('Hello World')