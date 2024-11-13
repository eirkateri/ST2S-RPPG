import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def analyze_results():
    mae_list = []
    rmse_list = []

    # Load the CSV file containing predictions and ground truth
    df = pd.read_csv(os.path.join(config['data_paths']['source_folder'], 'results.csv'))

    # Calculate MAE (Mean Absolute Error)
    df['MAE'] = abs(df['Ground Truth'] - df['Prediction'])
    mae = df['MAE'].mean()
    mae_list.append(mae)

    # Calculate RMSE (Root Mean Squared Error)
    df['RMSE'] = np.sqrt(((df['Prediction'] - df['Ground Truth']) ** 2).mean())
    rmse = df['RMSE'].mean()
    rmse_list.append(rmse)

    # Print the evaluation metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Optionally, calculate and print the Pearson correlation coefficient
    pearson_corr, _ = pearsonr(df['Ground Truth'], df['Prediction'])
    print(f"Pearson Correlation Coefficient: {pearson_corr}")

if __name__ == "__main__":
    analyze_results()