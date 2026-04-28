# main.py
from argparse import ArgumentParser
from src.components.ingestion import Ingestion
from src.components.validation import Validation
from src.components.transformation import TelecomGridTransformer
from src.components.model_trainer import ModelTrainer # New Import

def main():
    print("Hello from dustats-rca-project!")
    parser = ArgumentParser(description="Parse ingestion filepath and load from date")
    parser.add_argument("--date", dest='log_date', required=True, help="The log date from which to load data.")
    parser.add_argument("--path", dest="artifact_path", required=True, help="The path where artifacts to be stored.")
    parser.add_argument("-w", "--workers", help="Number of worker threads to use.", type=int, default=2)
    args = parser.parse_args()

    # Ingestion
    ingestion_obj = Ingestion(args.artifact_path)
    ingestion_artifact = ingestion_obj.ingest_data(args.log_date, args.workers)

    # Validation
    validation_obj = Validation(ingestion_artifact, args.artifact_path)
    validation_artifact, validation_status = validation_obj.validate_data()

    # Transformation
    transformation_obj = TelecomGridTransformer(validation_artifact, validation_status, args.artifact_path)
    transformation_artifact = transformation_obj.transform_and_save()

    # Training
    if transformation_artifact:
        trainer = ModelTrainer(transformation_artifact, args.artifact_path)
        model_artifact = trainer.initiate_model_training(epochs=100)
        print(f"Pipeline complete. Model at: {model_artifact}")

if __name__ == "__main__":
    main()