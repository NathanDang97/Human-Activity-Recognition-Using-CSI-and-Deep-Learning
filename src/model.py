import torch
import torch.nn as nn

# from grid search in the notebook
DROP_OUT = 0.2

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()
        # CNN extracts local time-frequency features from CSI data
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(DROP_OUT),
            nn.Conv1d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(DROP_OUT)
        )
        # BiLSTM captures sequence dependencies forward and backward
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=hidden_size, 
            batch_first=True,
            bidirectional=True
        )
        # Final FC layer maps hidden features to activity classes
        self.fc = nn.Linear(hidden_size * 2, num_classes) # double the hidden size since we set bidirectional to True

    def forward(self, x):
        x = x.permute(0, 2, 1)         # (B, F, T)
        x = self.cnn(x)                # (B, 256, T)
        x = x.permute(0, 2, 1)         # (B, T, 256)
        _, (h_n, _) = self.lstm(x)     # h_n: (4, B, H)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1) # last forward & backward
        return self.fc(h_cat) # (B, C)