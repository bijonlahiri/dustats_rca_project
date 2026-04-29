# main.py
import os
import sys
from argparse import ArgumentParser
from logger.logger import logging
from src.pipelines.training_pipeline import TrainingPipeline

def main():
    print("="*50)
    print("STARTING DUSTATS-RCA-PROJECT PIPELINE")
    print("="*50)
    
    parser = ArgumentParser(description="Parse ingestion filepath and load from date")
    parser.add_argument("--date", dest='log_date', required=True, help="The log date from which to load data.")
    parser.add_argument("--path", dest="artifact_path", required=True, help="The path where artifacts to be stored.")
    parser.add_argument("-w", "--workers", help="Number of worker threads to use.", type=int, default=2)
    parser.add_argument("--epochs", dest="epochs", help="The number of epochs for model training.", type=int, default=100)
    args = parser.parse_args()

    print(f"[INFO] Log Date: {args.log_date}")
    print(f"[INFO] Artifact Path: {args.artifact_path}")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] Epochs: {args.epochs}")

    try:
        # Initialize the pipeline
        print("\n[STEP] Initializing Training Pipeline...")
        pipeline = TrainingPipeline(args.artifact_path)

        # Execute the pipeline
        print("[STEP] Running Pipeline Components (Ingestion -> Validation -> Transformation -> Training)...")
        model_artifact_path = pipeline.run_pipeline(
            log_date=args.log_date, 
            workers=args.workers, 
            epochs=args.epochs
        )

        if model_artifact_path:
            print("\n" + "="*50)
            print("PIPELINE EXECUTION SUCCESSFUL")
            print(f"Model Artifacts saved at: {model_artifact_path}")
            print("="*50)
        else:
            print("\n[ERROR] Pipeline completed but no model artifact was generated.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[CRITICAL] Pipeline failed with error: {e}")
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()