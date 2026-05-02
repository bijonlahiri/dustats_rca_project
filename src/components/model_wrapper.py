import torch
import mlflow
import joblib
import pandas as pd
import numpy as np
from utils.utils import process_sessions
from src.components.model import MultiHeadLSTM

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
        self.index_cols = ['site_name', 'log_date', 'cellid', 'ueid', 'uptime']
        self.seq_len = 960
        self.max_uptime = 28770
        self.resolution = 30

    def predict(self, df):
        scaled_data = self.preprocessor.transform(df)
        # Reconstruct DF to keep track of columns after scaling
        # Note: ColumnTransformer reorders columns: [scaled_features..., remainder...]
        all_cols = self.feature_cols + [c for c in df.columns if c not in self.feature_cols]
        df = pd.DataFrame(scaled_data, columns=all_cols)
        results = df.groupby(by=self.index_cols).head(1)
        # 2. Process sessions
        X = process_sessions(
            df=df,
            feature_cols=self.feature_cols,
            index_cols=self.index_cols,
            seq_len=self.seq_len,
            max_uptime=self.max_uptime,
            resolution=self.resolution,
            return_y=False
        )
        start_time_pred, rca_pred = self.model(X)

        rca_dict = {
            0: 'BLER within limits',
            1: 'Degraded BLER due to poor channel conditions',
            2: 'Static BLER, channel is good',
            3: 'Scheduler limited MCS'
        }

        results['predicted_start_time'] = torch.mul(torch.mul(start_time_pred, 960).to(torch.int), 30).numpy().map(rca_dict)
        results['predicted_rca'] = torch.argmax(rca_pred, dim=-1).numpy()

        return results