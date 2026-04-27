from argparse import ArgumentParser
from src.components.ingestion import Ingestion

def main():
    print("Hello from dustats-rca-project!")
    parser = ArgumentParser(description="Parse ingestion filepath and load from date")
    parser.add_argument("--date", dest='log_date', required=True, help="The log date from which to load data from databricks table.")
    parser.add_argument("--path", dest="ingestion_filepath", required=True, help="The path where ingested data to be stored.")
    parser.add_argument("-w", "--workers", help="Number of worker threads to use.", type=int, default=2)
    args = parser.parse_args()
    ingestion_obj = Ingestion(args.ingestion_filepath)
    ingestion_obj.ingest_data(args.log_date, args.workers)


if __name__ == "__main__":
    main()
