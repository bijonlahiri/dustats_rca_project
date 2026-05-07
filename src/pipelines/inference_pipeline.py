import mlflow
import pandas as pd
from utils.utils import fetch_data_for_inference, process_sessions
from logger.logger import logging
from dotenv import load_dotenv
import torch

load_dotenv()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc')

_PREPROCESSOR_URI = 'models:/du_stats.training_data.preprocessor_model/3'
_LSTM_URI = 'models:/du_stats.training_data.multi_head_lstm_telecom_model/3'
_preprocessor = None
_lstm_model = None


def _load_models():
    global _preprocessor, _lstm_model
    if _preprocessor is None:
        _preprocessor = mlflow.sklearn.load_model(_PREPROCESSOR_URI)
    if _lstm_model is None:
        _lstm_model = mlflow.pytorch.load_model(_LSTM_URI, map_location=torch.device('cpu'))


class InferencePipeline:

    def __init__(self):
        _load_models()
        self.preprocessor = _preprocessor
        self.lstm_model = _lstm_model
        self.feature_cols = ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"]
        self.index_cols = ['site_name', 'log_date', 'cellid', 'ueid', 'uptime']
    
    def fetch_data(self, log_date:str, site_name:str, cellid:int=None, ueid:int=None)->pd.DataFrame:
        try:
            df = fetch_data_for_inference(
                log_date=log_date,
                site_name=site_name,
                cellid=cellid,
                ueid=ueid
            )
            num_rows = len(df)
            logging.info(f"Fetched data frame rows: {num_rows}")

            return df
        except Exception as e:
            logging.error(f"Failed to fetch data: {e}")
    
    def predict(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            feature_df = df.drop(['issue_start', 'rca_label'], axis=1)
            scaled_data = self.preprocessor.transform(feature_df)
            # all_cols = self.feature_cols + [c for c in feature_df.columns if c not in self.feature_cols]
            feature_df = pd.DataFrame(data=scaled_data, columns=self.preprocessor.get_feature_names_out())
            X = process_sessions(
                df=feature_df,
                feature_cols=self.feature_cols,
                index_cols=self.index_cols
            )
            df = df.set_index(self.index_cols).sort_index()
            df = df.groupby(level=[0, 1, 2, 3]).first()
            df = df.drop(self.feature_cols, axis=1)
            df = df.drop(['session_start', 'session_end', 'session_duration', 'session_id'], axis=1)
            y_start, y_rca = self.lstm_model(X)
            df['predicted_issue_start'] = ((y_start*960).to(torch.int)*30).numpy()
            df['predicted_rca'] = torch.argmax(torch.softmax(y_rca, dim=-1), dim=-1).numpy()

            return df
        except Exception as e:
            logging.error(f"Failed to predict: {e}")