from utils.utils import query_database, fetch_data
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import datetime
import pandas as pd
from argparse import ArgumentParser

class Ingestion:

    def __init__(self, ingestion_filepath:str):
        self.ingestion_filepath = ingestion_filepath
    
    def ingest_data(self, ingest_from_date:str, max_workers:int)->str:
        query = f""" SELECT DISTINCT log_date, site_name FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date > '{ingest_from_date}'
        """
        query_list = [{"log_date": datetime.datetime.strftime(result[0], format='%Y-%m-%d'), "site_name": result[1]} for result in query_database(query)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fetch_data, i, **item) for i, item in enumerate(query_list)]
            wait(futures)
            for result in as_completed(futures):
                _, log_date, site_name, num_rows = result.result()
                print(f"Fetched data for {log_date, site_name}, num_rows: {num_rows}")