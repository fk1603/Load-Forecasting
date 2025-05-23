#  Load Forecasting Project

This project trains and applies three different forecasting models for the purpose of load forecasting

## Project Structure

- `merge_load_files.py` – Merges multiple raw data files containing load data from ENTSO-E into one CSV file.  
  All used data is present in this repository under the `LoadData` folder.

- `LoadDataPreprocessing.py` – Preprocesses load data from raw CSVs, processes the merged load data file,  
  and prints final load CSV used later on when modeling.

- `WeatherDataPreprocessing.py` – Processes and interpolates weather data and writes this data to new CSV files.  
  All raw data is available in this repository.

- `compute_weighted_averages.py` – Computes weighted average weather covariate series and exports CSV files used later on when modeling.

- `NSM-ENTSOE.ipynb` – Notebook containing a Naive Seasonal Model and ENTSO-E forecast.  
  Used for benchmarks and model comparison.

- `RFR.ipynb` – Random Forest model definition, training and evaluation.

- `XGBoost_2.ipynb` – Extreme Gradient Boosting model definition, training and evaluation.

- `FINALTFT.ipynb` – Temporal Fusion Transformer model definition, training and evaluation.

- `create_boxplots.py` - Generates box plots for model comparisons

## How to Use

1. **(Optional) Run Data Preprocessing Scripts**

    Run these Python scripts in this order to prepare all data:

    `merge_load_files.py`
    `LoadDataPreprocessing.py`
    `WeatherDataPreprocessing.py`
    `compute_weighted_averages.py`
    
    Or import the already processed data when running the Jupyter Notebooks containing the models

2. **Open and run Jupyter Notebooks**

    Open and run the following notebook to generate benchmarks
        - `NSM-ENTSOE.ipynb`
   
    Open and run the following notebooks to train/import models and evaluate them
        - `RFR.ipynb`
        - `XGBoost_2.ipynb`
        - `FINALTFT.ipynb`

 3. **Compare results**

    Run `create_boxplots.py` to generate box plots used for model comparison


## Notes

- Trained XGB and TFT models have been saved and uploaded as `.pkl` or `.pt`. However, the RFR model
  doesn't have an uploaded saved model due to its large size (around 2.5 GB). Instead it's defined with a fixed random seed, so retraining
  won't change the model output. It will however take a couple of minutes to run `RFR.ipynb`

---

## Requirements

Install all required python packages as:

```bash
pip install -r requirements.txt
