import torch
import mlflow
import joblib
import pandas as pd
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
        # Create the reference backbone: [0, 30, 60, ..., 28770]
        self.reference_uptime = np.arange(0, 28770 + 30, 30)

    def predict(self, df):
        scaled_data = self.preprocessor.transform(df)
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

        rca_dict = {
            0: 'BLER within limits',
            1: 'Degraded BLER due to poor channel conditions',
            2: 'Static BLER, channel is good',
            3: 'Scheduler limited MCS'
        }

        results['predicted_start_time'] = torch.mul(torch.mul(start_time_pred, 960).to(torch.int), 30).numpy().map(rca_dict)
        results['predicted_rca'] = torch.argmax(rca_pred, dim=-1).numpy()

        return results