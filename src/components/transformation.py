import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from logger.logger import logging

class TelecomGridTransformer:
    def __init__(self, validation_artifact, validation_status, articact_path:str, max_uptime=28770, resolution=30):
        self.validation_artifact = validation_artifact
        self.validation_status = validation_status
        self.transformation_artifact = os.path.join(articact_path, "transformation/")
        os.makedirs(self.transformation_artifact, exist_ok=True)
        self.feature_cols = ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"]
        # Create the reference backbone: [0, 30, 60, ..., 28770]
        self.reference_uptime = np.arange(0, max_uptime + resolution, resolution)
        self.num_steps = len(self.reference_uptime)
        self.preprocessor = None

    def transform_and_save(self):
        if self.validation_status:
            df = pd.read_csv(self.validation_artifact)
            # 1. Fit/Transform Features before the join to avoid scaling zeros
            if self.preprocessor is None:
                self.preprocessor = ColumnTransformer(
                    transformers=[('num', StandardScaler(), self.feature_cols)],
                    remainder='passthrough'
                )
                # We keep session_id, rca_label, etc. via passthrough for now
                scaled_data = self.preprocessor.fit_transform(df)
                
                # Reconstruct DF to keep track of columns after scaling
                # Note: ColumnTransformer reorders columns: [scaled_features..., remainder...]
                all_cols = self.feature_cols + [c for c in df.columns if c not in self.feature_cols]
                df = pd.DataFrame(scaled_data, columns=all_cols)

            # 2. Prepare Reference DataFrame
            ref_df = pd.DataFrame({'uptime': self.reference_uptime})

            all_sessions_x = []
            all_rca_y = []
            all_start_y = []

            pd.set_option('future.no_silent_downcasting', True)

            # 3. Join each session to the reference grid
            for session_id, group in df.groupby('session_id'):
                # Join session data to the backbone
                # This automatically inserts rows for missing uptimes
                session_grid = pd.merge(ref_df, group, on='uptime', how='left')
                session_grid = session_grid.fillna(0).infer_objects(copy=False)
                
                # Ensure we only take the 960 steps (in case of data noise)
                session_grid = session_grid.sort_values('uptime').head(self.num_steps)
                
                # Extract features (X)
                x_tensor = torch.tensor(session_grid[self.feature_cols].values, dtype=torch.float32)
                all_sessions_x.append(x_tensor)
                
                # Targets (Y): Get the last non-zero label or the session's overall label
                # Using max() here assuming issue_start/rca_label are constant or we want the peak
                all_rca_y.append(group['rca_label'].iloc[-1]) 
                all_start_y.append(group['issue_start'].iloc[-1])

            # 4. Convert to PyTorch Tensors
            X = torch.stack(all_sessions_x)  # Shape: (Sessions, 960, 6)
            
            # Label encode the targets for the multi-head output
            # (Assuming you want these as numerical tensors)
            Y_rca = torch.tensor(pd.Series(all_rca_y).astype('category').cat.codes.values, dtype=torch.long)
            Y_start = torch.tensor(np.array(all_start_y), dtype=torch.float32)

            self._save_artifacts(X, Y_rca, Y_start)

            return self.transformation_artifact
        else:
            logging.warning(f"validation not succeded, skipping data transformation")

    def _save_artifacts(self, X, Y_rca, Y_start):
        # 80/20 Train/Test Split
        split = int(0.8 * len(X))
        
        torch.save({'x': X[:split], 'y_rca': Y_rca[:split], 'y_start': Y_start[:split]}, os.path.join(self.transformation_artifact, 'train.pth'))
        torch.save({'x': X[split:], 'y_rca': Y_rca[split:], 'y_start': Y_start[split:]}, os.path.join(self.transformation_artifact, 'test.pth'))
        joblib.dump(self.preprocessor, os.path.join(self.transformation_artifact, 'pre_processor.pkl'))
        
        print(f"Artifacts saved. Tensor Shape: {X.shape}")

# Usage:
# transformer = TelecomGridTransformer()
# transformer.transform_and_save(validation_artifact_df)