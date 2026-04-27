import os
from dotenv import load_dotenv
from databricks.sql import connect
from tqdm.auto import tqdm
from typing import List
from logger.logger import logging
from datetime import date
from tqdm.notebook import tqdm

def query_database(sql_query:str)->List:
    try:
        load_dotenv(override=True)
        with connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
        
        return rows
    except Exception as e:
        logging.error(f"Could not query database: {e}")

def fetch_data(tqdm_position, log_date:date, site_name:str):
    try:
        rows = []
        num_rows_query = f"""SELECT COUNT(*) FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND site_name = '{site_name}'
        """
        num_rows = query_database(num_rows_query)[0][0]
        batch_size = min(1, int(num_rows*0.05))
        query = f"""SELECT * FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND site_name = '{site_name}'
        """
        with tqdm(total=num_rows, desc=f"Fetching rows for {log_date, site_name}...", unit="rows", leave=True, position=tqdm_position) as pbar:
            with connect(
                server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                http_path=os.getenv("DATABRICKS_HTTP_PATH"),
                access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    for _ in range(num_rows // batch_size + 1):
                        row = cursor.fetchmany(batch_size)
                        rows.append(row)
                        pbar.update(batch_size)
        return rows, log_date, site_name, num_rows
    except Exception as e:
        logging.info(f"Could not fetch data for {log_date, site_name}: {e}")
    
if __name__=="__main__":
    log_date = '2026-01-01'
    site_name = "Nashik"
    value = fetch_data(log_date, site_name)
    print(value)