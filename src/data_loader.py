import pandas as pd

# Loads raw CSI data and corresponding labels from preprocessed CSV files.
# Drop the last sample to fix known bug in dataset
def load_raw_csi_data(data_path="data/data.csv", label_path="data/label.csv"):
    df_data = pd.read_csv(data_path, header=None)
    df_label = pd.read_csv(label_path, header=None)
    
    X_raw = df_data.values[:-1] # the last data point was reported to be a bug
    y_raw = df_label.values[:, 1]
    
    return X_raw, y_raw