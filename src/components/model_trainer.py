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
from src.components.model import MultiHeadLSTM
from src.components.model_wrapper import TelecomModelWrapper

class ModelTrainer:
    def __init__(self, transformation_artifact, artifact_path):
        self.artifact_path = artifact_path
        self.transformation_artifact = transformation_artifact
        self.model_path = os.path.join(artifact_path, "model/")
        os.makedirs(self.model_path, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"GPU available: {self.device==torch.device('cuda')}")
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
                df=pd.read_parquet(os.path.join(self.artifact_path, 'validation/validated_data'), engine='pyarrow'),
                source="s3://bijon-bucket/training_data/synth_time_series_rca_table/",
                targets="rca_label",
                name="Time series RCA dataset"
            )
            logging.info(f"Created dataset for mlflow logging")

            # Load transformed data
            train_data = torch.load(os.path.join(self.transformation_artifact, 'train.pth'))
            train_loader = DataLoader(
                TensorDataset(train_data['x'], train_data['y_start'], train_data['y_rca']), 
                batch_size=batch_size, shuffle=True
            )
            logging.info(f"Created train dataloader of size: {len(train_loader)}")

            test_data = torch.load(os.path.join(self.transformation_artifact, 'test.pth'))
            test_loader = DataLoader(
                TensorDataset(test_data['x'], test_data['y_start'], test_data['y_rca']),
                batch_size=batch_size, shuffle=False
            )
            logging.info(f"Created test data loader of size: {len(test_loader)}")

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
            logging.info(f"Model created with total parameters: {sum([p.numel() for p in model.parameters()])}")
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            logging.info(f"Optimizer created")

            with mlflow.start_run() as run:
                mlflow.log_params(params)

                # Create a dummy input and get predictions to infer the model signature
                input_example = torch.rand((32, 960, 6))
                logging.info(f"Input example created of size: {input_example.shape}")
                model.eval()
                with torch.inference_mode():
                    example_prediction = model(input_example.to(self.device))
                output_dict = {
                    "start_time_prediction": example_prediction[0].cpu().numpy(),
                    "rca_label_prediction": example_prediction[1].cpu().numpy()
                }
                for k, v in output_dict.items():
                    logging.info(f"Output dictionary key: {k}, size: {len(v)}")
                # Infer the signature
                signature = infer_signature(input_example.numpy(), output_dict)
                logging.info(f"Signature inferred")
                mlflow.log_input(dataset=mlflow_dataset, context="training")
                logging.info(f"MLFlow input logged.")

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
                    # signature=signature
                )
                print(f"Model trained and saved at {self.model_path}")
            return self.model_path

        except Exception as e:
            logging.error(f"Error in training: {e}")
            raise e