import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from data_loader import load_raw_csi_data
from preprocessing import preprocess_csi_data, train_test_val_split
from model import CNNLSTM
from train_eval import train_model, evaluate_model


# define the CSIDataset Class for NN training via PyTorch
class CSIDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), f"Mismatch between number of samples {len(X)} and labels {len(y)}"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# method to parse arguments from terminal   
def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on CSI Data")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM hidden dimension')
    return parser.parse_args()

# work pipeline
def main():
    # 0. Load the arguments from terminal
    args = parse_args()

    # 1. Load and preprocess data
    X, y = load_raw_csi_data()
    X_processed, y_processed, label_encoder = preprocess_csi_data(X, y)

    # 2. Train-val-test split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(X_processed, y_processed)

    # 3. Define the model
    # 3.1 training configs and hyperparameters from the best model from the notebook
    INPUT_SIZE = X_processed.shape[2] # no. subcarries
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs
    HIDDEN_DIM = args.hidden_dim
    NUM_CLASSES = len(label_encoder.classes_)
    # 3.2 hardware config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 3.3 model definition
    cnn_lstm_model = CNNLSTM(input_size=INPUT_SIZE, 
                             hidden_size=HIDDEN_DIM, 
                             num_classes=NUM_CLASSES)

    # 4. Create Dataloader
    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CSIDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(CSIDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # 5. Train and Evaluate the model
    train_model(cnn_lstm_model, train_loader, val_loader, device, 
                learning_rate=LEARNING_RATE, 
                epochs=NUM_EPOCHS)
    acc = evaluate_model(cnn_lstm_model, test_loader, device, True, label_encoder.classes_)
    print(f"Accuracy: {acc}")

# Note: see the notebook for visualization and fine-tuning
if __name__ == "__main__":
    main()