from src.pipelines.inference_pipeline import InferencePipeline
import pandas as pd
from argparse import ArgumentParser
import os
from logger.logger import logging

def rca_inference(log_date:str, site_name:str, cellid:int=None, ueid:int=None)->pd.DataFrame:
    try:
        inference_pipeline = InferencePipeline()
        df = inference_pipeline.fetch_data(
            log_date=log_date,
            site_name=site_name,
            cellid=cellid,
            ueid=ueid
        )
        predicted_df = inference_pipeline.predict(df)

        return predicted_df
    except Exception as e:
        logging.error(f"Failed to run inference: {e}")

if __name__=="__main__":
    try:
        parser = ArgumentParser(description="Arguments for inference: log_date, site_name, cellid, ueid")
        parser.add_argument("--path", dest="inference_path", required=True, help="Path to store the inferred data frame")
        parser.add_argument("--log_date", dest="log_date", required=True, help="Required argument")
        parser.add_argument("--site_name", dest="site_name", required=True, help="Required argument")
        parser.add_argument("--cellid", dest="cellid", default=None, help="Optional argument")
        parser.add_argument("--ueid", dest="ueid", default=None, help="Optional argument")

        args = parser.parse_args()

        storage_path = args.inference_path
        log_date = args.log_date
        site_name = str(args.site_name).lower()
        cellid = args.cellid
        ueid = args.ueid

        print(f"[INFO] Inference path: {storage_path}")
        print(f"[INFO] Log Date: {log_date}")
        print(f"[INFO] Site Name: {site_name}")
        print(f"[INFO] Cell Id: {cellid}")
        print(f"[INFO] UE Id: {ueid}")

        df = rca_inference(
            log_date=log_date,
            site_name=site_name,
            cellid=cellid,
            ueid=ueid
        )

        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        df.to_csv(os.path.join(storage_path, 'inference.csv'), index=True, header=True)

        print(f"[INFO] Inference run successfully and inferred data frame saved in path: {storage_path}")

    except Exception as e:
        logging.error(f"Failed to run inference pipeline: {e}")