import mlflow
import pandas as pd
from utils.utils import fetch_data_for_inference, process_sessions
from torch.utils.data import DataLoader, TensorDataset

class InferencePipeline:

    def __init__(self):
        mlflow.set_registry_uri('databricks-uc')
        self.preprocessor_model_uri = 'models:/du_stats.training_data.preprocessor_model/latest'
        self.lstm_model_uri = 'models:/multi_head_lstm_telecom_model/latest'
        self.preprocessor = mlflow.sklearn.load_model(self.preprocessor_model_uri)
        self.lstm_model = mlflow.pytorch.load_model(self.lstm_model_uri)
        self.feature_cols = ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"]
        self.index_cols = ['site_name', 'log_date', 'cellid', 'ueid', 'uptime']
    
    def fetch_data(self, log_date:str, site_name:str, cellid:int=None, ueid:int=None)->pd.DataFrame:
        df = fetch_data_for_inference(
            log_date=log_date,
            site_name=site_name,
            cellid=cellid,
            ueid=ueid
        )

        df = df.drop(['issue_start', 'rca_label'], axis=1)

        return df
    
    def predict(self, df:pd.DataFrame)->pd.DataFrame:
        scaled_data = self.preprocessor.transform(df)
        all_cols = self.feature_cols + [c for c in df.columns if c not in self.feature_cols]
        df = pd.DataFrame(data=scaled_data, columns=all_cols)
        X = process_sessions(
            df=df,
            feature_cols=self.feature_cols,
            index_cols=self.index_cols
        )
        y_start, y_rca = self.lstm_model(X)
        df['predicted_issue_start'] = y_start
        df['predicted_rca'] = y_rca

        return df