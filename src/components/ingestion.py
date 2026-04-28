from utils.utils import query_database, fetch_data
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import pandas as pd
from logger.logger import logging
import os

class Ingestion:

    def __init__(self, artifact_filepath:str):
        self.ingestion_filepath = os.path.join(artifact_filepath, "ingestion/ingestion.csv")
        os.makedirs(os.path.dirname(self.ingestion_filepath), exist_ok=True)
        self.write_lock = Lock()
    
    def ingest_data(self, ingest_from_date:str, max_workers:int)->str:
        query = f""" SELECT DISTINCT log_date, site_name FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date > '{ingest_from_date}'
        """
        query_list = [
            {
                "log_date": str(result[0]),
                "site_name": result[1],
            } for result in query_database(query)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            logging.info(f"Executing {len(query_list)} fetch tasks.")
            futures = [executor.submit(fetch_data, **item) for item in query_list]
            ingestion_file_exists = os.path.isfile(self.ingestion_filepath)
            for result in as_completed(futures):
                row_df = result.result()
                # Use a lock to ensure only one thread writes at a time
                with self.write_lock:
                    # Check existence right now, not at the start of the function
                    file_exists = os.path.isfile(self.ingestion_filepath)
                    # Check if file is empty (size 0) to be extra safe
                    has_content = file_exists and os.path.getsize(self.ingestion_filepath) > 0
                    
                    row_df.to_csv(
                        self.ingestion_filepath, 
                        mode='a', 
                        index=False, 
                        header=not has_content
                    )
        return self.ingestion_filepath