import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reshape each sample into (subcarriers, time_steps) and normalize
# Add simple statistical features (mean, std, min, max, energy) for augmentation
def preprocess_csi_data(X_raw, y_raw, num_subcarriers=114, smoothing_window=3, augment=False):
    num_samples, num_features = X_raw.shape
    if num_features % num_subcarriers != 0:
        raise ValueError(f"Feature count {num_features} is not divisible by {num_subcarriers}.")

    time_steps = num_features // num_subcarriers
    
    # reshaping to (N, F, T) where N = no. of samples, T = n
    X = X_raw.reshape((num_samples, num_subcarriers, time_steps))

    # transpose to (N, T, F)
    X = np.transpose(X, (0, 2, 1))

    # Normalize each sample across subcarriers and time
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-6)

    # Smoothing
    padded = np.pad(X, ((0, 0), (smoothing_window//2, smoothing_window//2), (0, 0)), mode='reflect')
    X = np.stack([np.mean(padded[:, i:i + smoothing_window, :], axis=1)
                  for i in range(X.shape[1])], axis=1)

    # Statistical features
    def extract_stats(x):
        mean = np.mean(x, axis=1)
        std = np.std(x, axis=1)
        min_ = np.min(x, axis=1)
        max_ = np.max(x, axis=1)
        energy = np.sum(np.square(x), axis=1)
        return np.stack([mean, std, min_, max_, energy], axis=1)

    # concatenate statistics along the time axis
    X_stats = extract_stats(X)
    print(f"Reshaping to (samples={num_samples}, time_steps and stats={time_steps + 5}, subcarriers={num_subcarriers})")
    X_aug = np.concatenate([X, X_stats], axis=1)  # shape: (N, T+5, F)

    # label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    return X_aug, y_encoded, label_encoder

# split the data to train, val, and test set
def train_test_val_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    print("Train / Val / Test data sizes:", len(X_train), len(X_val), len(X_test))
    print("Train / Val / Test label sizes:", len(y_train), len(y_val), len(y_test))
    
    return X_train, X_val, X_test, y_train, y_val, y_test