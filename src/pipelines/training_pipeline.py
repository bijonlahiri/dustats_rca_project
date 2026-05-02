import os
import sys
from logger.logger import logging
from src.components.ingestion import Ingestion
from src.components.validation import Validation
from src.components.transformation import TelecomGridTransformer
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self, artifact_path: str):
        """
        Initializes the pipeline with a root path for all artifacts.
        """
        self.artifact_path = artifact_path
        # Clean existing artifacts if necessary
        if os.path.exists(self.artifact_path):
            import shutil
            shutil.rmtree(self.artifact_path)
            logging.info(f"Cleaned existing artifact directory: {self.artifact_path}")

    def run_pipeline(self, log_date: str, workers: int, epochs: int, tqdm_disable:bool=True):
        """
        Orchestrates the execution of the entire training lifecycle.
        """
        try:
            logging.info("Starting Pipeline Execution...")

            # 1. Ingestion
            logging.info("Step 1: Data Ingestion")
            ingestion = Ingestion(self.artifact_path)
            ingestion_artifact = ingestion.ingest_data(log_date, workers, tqdm_disable)

            # 2. Validation
            logging.info("Step 2: Data Validation")
            validation = Validation(ingestion_artifact, self.artifact_path)
            validation_artifact, validation_status = validation.validate_data()

            # 3. Transformation
            logging.info("Step 3: Data Transformation")
            transformation = TelecomGridTransformer(
                validation_artifact, 
                validation_status, 
                self.artifact_path
            )
            transformation_artifact = transformation.transform_and_save()

            # 4. Model Training
            if transformation_artifact:
                logging.info("Step 4: Model Training")
                trainer = ModelTrainer(transformation_artifact, self.artifact_path)
                model_artifact = trainer.initiate_model_training(
                    epochs=epochs,
                    batch_size=32,
                    learning_rate=1e-4,
                    catalog='du_stats',
                    schema='training_data',
                    model_name='multi_head_lstm_telecom_rca_model'
                )
                logging.info(f"Pipeline completed successfully. Model saved at: {model_artifact}")
                return model_artifact
            else:
                logging.error("Pipeline aborted: Transformation artifact was not created.")
                return None

        except Exception as e:
            logging.error(f"Pipeline failed due to error: {str(e)}")
            raise e

if __name__ == "__main__":
    # Example local test run
    pipeline = TrainingPipeline(artifact_path="artifacts/test_run")
    pipeline.run_pipeline(log_date="2023-01-01", workers=4, epochs=10)