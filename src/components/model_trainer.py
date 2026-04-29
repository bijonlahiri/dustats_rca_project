# src/components/model_trainer.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from logger.logger import logging
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from utils.utils import train_step, validation_step

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

class ModelTrainer:
    def __init__(self, transformation_artifact, artifact_path):
        self.artifact_path = artifact_path
        self.transformation_artifact = transformation_artifact
        self.model_path = os.path.join(artifact_path, "model/")
        os.makedirs(self.model_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/bijonlahiri@gmail.com/multi_head_lstm")
        mlflow.set_registry_uri('databricks-uc')

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

    def initiate_model_training(
            self,
            epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            catalog="main",
            schema="default",
            model_name="telecom_rca_model"):
        try:

            mlflow_dataset = mlflow.data.from_pandas(       # For mlflow dataset logging
                df=pd.read_csv(os.path.join(self.artifact_path, 'validation/validated_data.csv')),
                source="s3://bijon-bucket/training_data/synth_time_series_rca_table/",
                targets="rca_label",
                name="Time series RCA dataset"
            )

            # Load transformed data
            train_data = torch.load(os.path.join(self.transformation_artifact, 'train.pth'))
            train_loader = DataLoader(
                TensorDataset(train_data['x'], train_data['y_start'], train_data['y_rca']), 
                batch_size=batch_size, shuffle=True
            )

            test_data = torch.load(os.path.join(self.transformation_artifact, 'test.pth'))
            test_loader = DataLoader(
                TensorDataset(test_data['x'], test_data['y_start'], test_data['y_rca']),
                batch_size=batch_size, shuffle=False
            )

            params = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "d_in": 6,
                "d_out": 64,
                "num_lstm_layers": 1,
                "shortcut": True
            }

            model = MultiHeadLSTM(
                d_in=params["d_in"],
                d_out=params["d_out"],
                num_lstm_layers=params["num_lstm_layers"],
                shortcut=params["shortcut"]
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

            with mlflow.start_run() as run:
                mlflow.log_params(params)

                # Create a dummy input and get predictions to infer the model signature
                input_example = torch.rand((32, 960, 6))
                model.eval()
                with torch.inference_mode():
                    example_prediction = model(input_example.to(self.device))
                output_dict = {
                    "start_time_prediction": example_prediction[0].cpu().numpy(),
                    "rca_label_prediction": example_prediction[1].cpu().numpy()
                }
                # Infer the signature
                signature = infer_signature(input_example.numpy(), output_dict)
                mlflow.log_params(params)
                mlflow.log_input(dataset=mlflow_dataset, context="training")

                model.train()
                for epoch in range(epochs):

                    train_loss, train_start_mae, train_rca_acc = train_step(model, train_loader, optimizer, self.device)
                    test_loss, test_start_mae, test_rca_acc = validation_step(model, test_loader, self.device)
    
                    if epoch % 5 == 0:
                        mlflow.log_metric("loss", train_loss, step=epoch)
                        mlflow.log_metric("start_mae", train_start_mae, step=epoch)
                        mlflow.log_metric("RCA accuracy", train_rca_acc, step=epoch)
                        mlflow.log_metric("test_loss", test_loss, step=epoch)
                        mlflow.log_metric("test_start_mae", test_start_mae, step=epoch)
                        mlflow.log_metric("test_RCA accuracy", test_rca_acc, step=epoch)
                    logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")

                full_model_name = f'{catalog}.{schema}.{model_name}'
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name='lstm_telecom_rca_model',
                    signature=signature
                )

            torch.save(model.state_dict(), os.path.join(self.model_path, "model.pth"))
            print(f"Model trained and saved at {self.model_path}")
            return self.model_path

        except Exception as e:
            logging.error(f"Error in training: {e}")
            raise e