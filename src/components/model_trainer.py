# src/components/model_trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from logger.logger import logging

class MultiHeadLSTM(nn.Module):
    def __init__(self, d_in=6, num_lstm_layers=1, bidirectional=False, d_out=32, hidden_size_linear=32, shortcut=False, dropout=0.0):
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
        self.shared_dense_2 = nn.Linear(d_out * (2**bidirectional_multiplier), hidden_size_linear)

        self.start_time_head = nn.Linear(hidden_size_linear, 1)
        self.rca_label_head = nn.Linear(hidden_size_linear, 4) # Assuming 4 classes

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

class ModelTrainer:
    def __init__(self, transformation_artifact, artifact_path):
        self.transformation_artifact = transformation_artifact
        self.model_path = os.path.join(artifact_path, "model/")
        os.makedirs(self.model_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _eval_loss(self, y_pred, y_true, mask, weights):
        y_start_pred = y_pred[0]
        y_rca_label_logits = y_pred[1]

        y_start_true = y_true[0]
        y_rca_label_true = y_true[1]

        y_start_loss_fn = torch.nn.MSELoss(reduction='none')
        y_rca_label_loss_fn = torch.nn.CrossEntropyLoss()

        y_start_loss = y_start_loss_fn(y_start_pred, y_start_true)
        y_rca_label_loss = y_rca_label_loss_fn(y_rca_label_logits, y_rca_label_true)

        masked_y_start_loss = (y_start_loss*mask).sum()/mask.sum()

        total_loss = masked_y_start_loss*weights[0] + y_rca_label_loss * weights[1]

        return total_loss

    def initiate_model_training(self, epochs=10, batch_size=32):
        try:
            # Load transformed data
            train_data = torch.load(os.path.join(self.transformation_artifact, 'train.pth'))
            train_loader = DataLoader(
                TensorDataset(train_data['x'], train_data['y_start'], train_data['y_rca']), 
                batch_size=batch_size, shuffle=True
            )

            model = MultiHeadLSTM(
                d_in=6,
                shortcut=True
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_start, batch_rca in train_loader:
                    batch_x, batch_start, batch_rca = batch_x.to(self.device), batch_start.to(self.device), batch_rca.to(self.device)
                    
                    optimizer.zero_grad()
                    pred_start, pred_rca = model(batch_x)

                    y_pred = (pred_start, pred_rca)
                    y_true = (batch_start, batch_rca)

                    mask = (batch_rca != 0)
                    
                    loss = self._eval_loss(y_pred, y_true, mask, (1, 0.1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

            torch.save(model.state_dict(), os.path.join(self.model_path, "model.pth"))
            print(f"Model trained and saved at {self.model_path}")
            return self.model_path

        except Exception as e:
            logging.error(f"Error in training: {e}")
            raise e