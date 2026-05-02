import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from logger.logger import logging
from utils.utils import process_sessions

class TelecomGridTransformer:
    def __init__(self, validation_artifact, validation_status, articact_path:str, seq_len=960, max_uptime=28770, resolution=30):
        self.validation_artifact = validation_artifact
        self.validation_status = validation_status
        self.transformation_artifact = os.path.join(articact_path, "transformation/")
        os.makedirs(self.transformation_artifact, exist_ok=True)
        self.feature_cols = ["cqi", "mcs", "ibler", "rbler", "resbler", "tbler"]
        self.index_cols = ['site_name', 'log_date', 'cellid', 'ueid', 'uptime']
        self.max_uptime = max_uptime
        self.resolution = resolution
        self.seq_len=seq_len
        self.preprocessor = None

    def transform_and_save(self):
        if self.validation_status:
            logging.info(f"Validation status passed.")
            df = pd.read_parquet(self.validation_artifact, engine="pyarrow")
            # 1. Fit/Transform Features before the join to avoid scaling zeros
            if self.preprocessor is None:
                self.preprocessor = ColumnTransformer(
                    transformers=[('num', StandardScaler(), self.feature_cols)],
                    remainder='passthrough'
                )
                # We keep session_id, rca_label, etc. via passthrough for now
                scaled_data = self.preprocessor.fit_transform(df)
                logging.info(f"Columns in column transformer: {self.preprocessor.get_feature_names_out()}")
                
                # Reconstruct DF to keep track of columns after scaling
                # Note: ColumnTransformer reorders columns: [scaled_features..., remainder...]
                all_cols = self.feature_cols + [c for c in df.columns if c not in self.feature_cols]
                df = pd.DataFrame(scaled_data, columns=all_cols)
                logging.info(f"Preprocessed and scaled: Total length: {len(df)}\t Total columns: {len(df.columns)}")

            # 2. Process sessions
            X, Y_start, Y_rca = process_sessions(
                df=df,
                feature_cols=self.feature_cols,
                index_cols=self.index_cols,
                seq_len=self.seq_len,
                max_uptime=self.max_uptime,
                resolution=self.resolution,
                return_y=True
            )

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