# src/components/model_trainer.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from logger.logger import logging
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import joblib
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
    
class TelecomModelWrapper(mlflow.pyfunc.PyFuncModel):
    def load_context(self, context):
        # Load the preprocessor
        self.preprocessor = joblib.load(context.artifacts['preprocessor'])

        # Load the pytorch model
        model_params = context.artifacts['model_params']
        self.model = MultiHeadLSTM(
            d_in=model_params['d_in'],
            d_out=model_params['d_out'],
            bidirectional=model_params['bidirectional'],
            num_lstm_layers=model_params['num_lstm_layers'],
            shortcut=model_params['shortcut'],
            dropout=model_params['dropout']
        )
        self.model.load_state_dict(torch.load(context.artifacts['lstm_model']))
        self.model.eval()
        self.feature_cols = ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"]
        # Create the reference backbone: [0, 30, 60, ..., 28770]
        self.reference_uptime = np.arange(0, 28770 + 30, 30)

    def predict(self, df):
        scaled_data = self.preprocessor.fit_transform(df)
        # Reconstruct DF to keep track of columns after scaling
        # Note: ColumnTransformer reorders columns: [scaled_features..., remainder...]
        all_cols = self.feature_cols + [c for c in df.columns if c not in self.feature_cols]
        df = pd.DataFrame(scaled_data, columns=all_cols)
        # 2. Prepare Reference DataFrame
        ref_df = pd.DataFrame({'uptime': self.reference_uptime})

        all_sessions_x = []
        results = []

        pd.set_option('future.no_silent_downcasting', True)

        # 3. Join each session to the reference grid
        for session_id, group in df.groupby(['site_name', 'log_date', 'cellid', 'ueid']):
            # Join session data to the backbone
            # This automatically inserts rows for missing uptimes
            session_grid = pd.merge(ref_df, group, on='uptime', how='left')
            session_grid = session_grid.fillna(0).infer_objects(copy=False)
            
            # Ensure we only take the 960 steps (in case of data noise)
            session_grid = session_grid.sort_values('uptime').head(self.num_steps)

            results.append(group.drop([self.feature_cols] + ['uptime']))
            
            # Extract features (X)
            x_tensor = torch.tensor(session_grid[self.feature_cols].values, dtype=torch.float32)
            all_sessions_x.append(x_tensor)
        # 4. Convert to PyTorch Tensors
        X = torch.stack(all_sessions_x)  # Shape: (Sessions, 960, 6)

        results = pd.concat(results)
        start_time_pred, rca_pred = self.model(X)

        results['predicted_start_time'] = start_time_pred.numpy()
        results['predicted_rca'] = rca_pred.numpy()

        return results


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
                "d_out": 128,
                "num_lstm_layers": 2,
                "shortcut": True,
                "bidirectional": False,
                "dropout": 0.0
            }

            model = MultiHeadLSTM(
                d_in=params["d_in"],
                d_out=params["d_out"],
                num_lstm_layers=params["num_lstm_layers"],
                shortcut=params["shortcut"],
                bidirectional=params['bidirectional'],
                dropout=params['dropout']
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
                # mlflow.pytorch.log_model(
                #     pytorch_model=model,
                #     name='lstm_telecom_rca_model',
                #     signature=signature
                # )
                preprocessor_model = joblib.load(os.path.join(self.artifact_path, 'transformation/pre_processor.pkl'))
                preprocessor_signature = infer_signature(input_example, preprocessor_model.transform(input_example))
                # mlflow.sklearn.log_model(
                #     sk_model=preprocessor_model,
                #     name='preprocessor_model',
                #     signature=preprocessor_signature
                # )
                joblib.dump(preprocessor_model, os.path.join(self.model_path, 'preprocessor.pkl'))
                torch.save(model.state_dict(), os.path.join(self.model_path, "model.pth"))
                model_artifacts = {
                    'preprocessor': 'preprocessor.pkl',
                    'lstm_model': 'model.pth',
                    'model_params': params
                }
                mlflow.pyfunc.log_model(
                    python_model=TelecomModelWrapper(),
                    name='telecom_rca_model',
                    artifacts=model_artifacts,
                    pip_requirements=['torch', 'scikit-learn', 'joblib'],
                    signature=signature
                )
                print(f"Model trained and saved at {self.model_path}")
            return self.model_path

        except Exception as e:
            logging.error(f"Error in training: {e}")
            raise e