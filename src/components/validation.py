import pandas as pd
import numpy as np
from scipy import stats
from logger.logger import logging
import os
import json

class Validation:
    def __init__(self, ingestion_artifact: str, artifact_filepath: str):
        self.ingestion_artifact = ingestion_artifact
        # Directory for all validation outputs
        self.validation_dir = os.path.join(artifact_filepath, "validation/")
        os.makedirs(self.validation_dir, exist_ok=True)
        
        self.df = None
        self.validation_status = True  # Flag to track if data is save-ready
        self.report_metrics = {}       # Dictionary to store report data

    def load_data(self):
        self.df = pd.read_csv(self.ingestion_artifact)
        logging.info("Data loaded successfully for validation.")
        return self.df

    def validate_structural_integrity(self):
        invalid_time = self.df[
            (self.df['issue_start'] < self.df['session_start']) | 
            (self.df['issue_start'] > self.df['session_end'])
        ]
        self.report_metrics['invalid_time_windows'] = len(invalid_time)
        
        if not invalid_time.empty:
            logging.warning(f"Found {len(invalid_time)} sessions with invalid time windows.")

        for col in ['ibler', 'rbler', 'tbler']:
            out_of_bounds = self.df[(self.df[col] < 0) | (self.df[col] > 100)]
            self.report_metrics[f'{col}_out_of_bounds_count'] = len(out_of_bounds)
            if not out_of_bounds.empty:
                out_of_bounds.to_csv(os.path.join(self.validation_dir, f'{col}_out_of_bounds.csv'), index=False)
                logging.error(f"Critical: {col} has values outside [0, 100]. Status set to Failed.")
                self.validation_status = False

    def validate_rf_physics(self):
        anomaly = self.df[(self.df['cqi'] > 12) & (self.df['mcs'] < 5)]
        self.report_metrics['physics_anomalies'] = len(anomaly)
        if not anomaly.empty:
            logging.info(f"Physics Warning: {len(anomaly)} rows show high CQI but low MCS.")

    def run_hypothesis_tests(self):
        logging.info("Running Kruskal-Wallis H-test...")
        h_results = {}
        features_to_test = ['cqi', 'mcs', 'ibler', 'tbler', 'rbler', 'resbler']
        for feature in features_to_test:
            groups = [group[feature].values for _, group in self.df.groupby('rca_label')]
            _, p_val = stats.kruskal(*groups)
            h_results[feature] = round(p_val, 4)
            logging.info(f"{feature} Significance: p={p_val:.4f}")
        self.report_metrics['hypothesis_tests_p_values'] = h_results

    def validate_lstm_readiness(self):
        self.df['expected_ticks'] = (self.df['session_duration'] / 30 + 1).astype(int)
        
        # Check sequence continuity
        actual_counts = self.df.groupby('session_id').size().reset_index(name='actual_ticks')
        validation_df = self.df[['session_id', 'expected_ticks']].drop_duplicates().merge(actual_counts, on='session_id')
        mismatched = validation_df[validation_df['expected_ticks'] != validation_df['actual_ticks']]
        
        self.report_metrics['mismatched_sequences'] = len(mismatched)
        if not mismatched.empty:
            mismatched.to_csv(os.path.join(self.validation_dir, "mismatched_sessions.csv"), index=False)
            logging.error(f"Critical: {len(mismatched)} sessions have sequence gaps.")
            self.validation_status = False

    def generate_report(self):
        """Saves a summary of the validation as a JSON file."""
        report_path = os.path.join(self.validation_dir, "validation_report.json")
        summary = {
            "status": "PASSED" if self.validation_status else "FAILED",
            "total_sessions": int(self.df['session_id'].nunique()),
            "total_rows": int(len(self.df)),
            "metrics": self.report_metrics
        }
        with open(report_path, 'w') as f:
            json.dump(summary, f)
        logging.info(f"Validation report saved to {report_path}")

    def validate_data(self):
        self.load_data()
        
        # Core metadata logging
        log_days = self.df['log_date'].unique().tolist()
        self.report_metrics['days_count'] = len(log_days)
        
        # Run all checks
        self.validate_structural_integrity()
        self.validate_rf_physics()
        self.validate_lstm_readiness()
        self.run_hypothesis_tests()
        
        # Generate the report
        self.generate_report()

        # Final Save Logic
        if self.validation_status:
            save_path = os.path.join(self.validation_dir, "validated_data.csv")
            self.df.to_csv(save_path, index=False)
            logging.info(f"SUCCESS: Data passed all critical checks. Saved to {save_path}")
        else:
            logging.error("FAILURE: Data failed critical validation. Validated CSV was not saved.")
        
        return save_path, self.validation_status