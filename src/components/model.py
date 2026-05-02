import torch
from torch import nn

class MultiHeadLSTM(nn.Module):
    def __init__(self, d_in=6, num_lstm_layers=1, bidirectional=False, d_out=32, shortcut=False, dropout=0.0):
        super().__init__()
        self.shortcut = shortcut
        # Adjusted d_in to 6 to match feature_cols in transformation.py
        self.expansion_layer = nn.Linear(d_in, d_out)

        self.lstm = nn.LSTM(
            input_size=d_out,
            hidden_size=d_out,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        bidirectional_multiplier = 1 if bidirectional else 0
        self.dropout = nn.Dropout(dropout)
        self.shared_dense_2 = nn.Linear(d_out * (2**bidirectional_multiplier), d_out)

        self.start_time_head = nn.Linear(d_out, 1)
        self.rca_label_head = nn.Linear(d_out, 4) # Assuming 4 classes

    def forward(self, x):
        x = self.expansion_layer(x)
        out, _ = self.lstm(x)
        
        if self.shortcut:
            out = x + out
        else:
            out = out
            
        last_hidden = torch.mean(out, dim=1)
        shared_2 = torch.relu(self.shared_dense_2(last_hidden))
        shared_2 = self.dropout(shared_2)

        if self.shortcut:
            shared_2 = shared_2 + last_hidden
        else:
            shared_2 = shared_2

        out_start_time = self.start_time_head(shared_2)
        out_rca_label = self.rca_label_head(shared_2)

        return out_start_time.squeeze(), out_rca_label