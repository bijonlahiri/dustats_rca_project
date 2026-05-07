import os
from dotenv import load_dotenv
from databricks.sql import connect
from typing import List
from logger.logger import logging
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import gc
import pyarrow.parquet as pq
import pyarrow as pa
from threading import Lock
from sklearn.metrics import precision_score, recall_score, f1_score

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

def fetch_data_for_inference(log_date:str, site_name:str, cellid:int=None, ueid:int=None)->pd.DataFrame:
    try:
        load_dotenv(override=True)
        logging.info(f"Fetching data for {site_name, log_date}")
        if not cellid and not ueid:
            fetch_query = f"""
        SELECT * FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND LOWER(site_name) = '{site_name}'
        """
        elif not ueid:
            fetch_query = f"""
        SELECT * FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND LOWER(site_name) = '{site_name}' AND cellid = '{cellid}'
        """
        else:
            fetch_query = f"""
        SELECT * FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND LOWER(site_name) = '{site_name}' AND cellid = '{cellid}' AND ueid = '{ueid}'
        """
        logging.info(f"Fetch query:\n {fetch_query}")
        batch_size = 1000
        batches = []

        with connect(
            server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(fetch_query)
                while True:
                    # fetchmany_arrow returns a pyarrow.Table
                    batch_table = cursor.fetchmany_arrow(batch_size)
                    if not batch_table:
                        break
                    batches.append(batch_table)
        logging.info(f"Total batch tables: {len(batches)}")
        full_table = pa.concat_tables(batches)

        return full_table.to_pandas()


    except Exception as e:
        logging.error(f"Error fetching data: {e}")

def fetch_data(log_date:str, site_name:str, output_path:str, tqdm_disable:bool=True):
    try:
        load_dotenv(override=True)
        writer = None
        logging.info(f"Fetching data for {site_name, log_date} with TQDM disable: {tqdm_disable}")
        df = pd.DataFrame()
        rows_query = f"""
        SELECT COUNT(*) FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND site_name = '{site_name}'
        """
        num_rows = query_database(rows_query)[0][0]
        batch_size = 1000 #min(1, int(num_rows*0.05))
        query = f"""SELECT * FROM `du_stats`.`training_data`.`synth_time_series_rca_table`
        WHERE log_date = DATE '{log_date}' AND site_name = '{site_name}'
        """
        with tqdm(total=num_rows, desc=f"Fetching data for {log_date, site_name}...", unit="row",disable=tqdm_disable) as pbar:
            with connect(
                server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                http_path=os.getenv("DATABRICKS_HTTP_PATH"),
                access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    logging.info(f"Writing to file: {output_path}")
                    while True:
                        # fetchmany_arrow returns a pyarrow.Table
                        batch_table = cursor.fetchmany_arrow(batch_size) 
                        
                        if batch_table and batch_table.num_rows > 0:
                            if writer is None:
                                # Initialize the writer with the schema of the first batch
                                writer = pq.ParquetWriter(output_path, batch_table.schema)
                            
                            # Write the arrow table directly to disk
                            writer.write_table(batch_table)
                            pbar.update(batch_table.num_rows)
                        else:
                            break

                    if writer:
                        writer.close()

        return 1
    except Exception as e:
        logging.info(f"Could not fetch data for {log_date, site_name}: {e}")

def accuracy_fn(y_logits: torch.Tensor, y_true: torch.Tensor)->float:
    y_pred = torch.argmax(y_logits, dim=1)

    return torch.eq(y_pred, y_true).sum().item()/len(y_true)*100

def precision_fn(y_logits: torch.tensor, y_true: torch.tensor)->float:
    y_pred = torch.argmax(torch.sigmoid(y_logits, dim=-1), dim=-1)
    y_pred, y_true = y_pred.cpu().numpy(), y_true.cpu().numpy()
    p_score = precision_score(y_true, y_pred)

    return p_score

def mae_eval(y_pred:torch.Tensor, y_true:torch.Tensor)->float:
    return torch.abs(y_pred - y_true).sum().item()/len(y_true)

def get_inverse_class_weights(y_label:torch.Tensor, num_classes:int, device:torch.device):
    class_weights = []
    for i in range(num_classes):
        class_samples = (y_label == i).sum()
        class_weights.append(len(y_label)/num_classes/class_samples)
    return torch.Tensor(class_weights).to(device)

def eval_loss(y_pred, y_true, mask, weights, device):

    y_start_pred = y_pred[0]
    y_rca_label_logits = y_pred[1]

    y_start_true = y_true[0]
    y_rca_label_true = y_true[1]

    class_weigths = get_inverse_class_weights(y_rca_label_true, 4, device)

    y_start_loss_fn = torch.nn.MSELoss(reduction='none')
    y_rca_label_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weigths)

    y_start_loss = y_start_loss_fn(y_start_pred, y_start_true)
    y_rca_label_loss = y_rca_label_loss_fn(y_rca_label_logits, y_rca_label_true)

    masked_y_start_loss = (y_start_loss*mask).sum()/mask.sum()

    total_loss = masked_y_start_loss*weights[0] + y_rca_label_loss * weights[1]

    return total_loss

def train_step(model:torch.nn.Module, train_loader, optimizer, device):
  model.to(device).train()
  train_loss = start_mae = rca_acc = 0
  for idx, (X_train, y_start_train, y_rca_label_train) in enumerate(train_loader):
    X_train = X_train.to(device)
    y_start_train = y_start_train.to(device)
    y_rca_label_train = y_rca_label_train.to(device)

    # Forward pass
    y_start_pred, y_rca_label_logits = model(X_train)
    mask = (torch.ones_like(y_rca_label_train))

    y_pred = (y_start_pred, y_rca_label_logits)
    y_true = (y_start_train, y_rca_label_train)

    loss = eval_loss(y_pred, y_true, mask, (100, 1), device)
    train_loss += loss.item()
    start_mae += mae_eval((y_start_pred*mask*960).to(torch.int)*30, (y_start_train*mask*960).to(torch.int)*30)
    rca_acc += accuracy_fn(y_rca_label_logits, y_rca_label_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return train_loss/len(train_loader), start_mae/len(train_loader), rca_acc/len(train_loader)

def validation_step(model:torch.nn.Module, test_loader, device):
  model.to(device).eval()
  loss = start_mae = rca_acc = 0
  with torch.inference_mode():
    for idx, (X_test, y_start_test, y_rca_label_test) in enumerate(test_loader):
      X_test = X_test.to(device)
      y_start_test = y_start_test.to(device)
      y_rca_label_test = y_rca_label_test.to(device)

      # Forward
      y_start_pred, y_rca_label_logits = model(X_test)
      mask = (torch.ones_like(y_rca_label_test))
      y_pred = (y_start_pred, y_rca_label_logits)
      y_true = (y_start_test, y_rca_label_test)

      loss += eval_loss(y_pred, y_true, mask, (1, 1), device)
      start_mae += mae_eval((y_start_pred*mask*960).to(torch.int)*30, (y_start_test*mask*960).to(torch.int)*30)
      rca_acc += accuracy_fn(y_rca_label_logits, y_rca_label_test)

  return loss.item()/len(test_loader), start_mae/len(test_loader), rca_acc/len(test_loader)

def process_sessions(
        df:pd.DataFrame,
        feature_cols:List,
        index_cols:List,
        seq_len:int=960,
        max_uptime:int=28770,
        resolution:int=30,
        return_y:bool=False):
    ref_time = pd.DataFrame(data=np.array(np.arange(0, max_uptime+resolution, resolution)), columns=['uptime'])
    ref_time.to_csv('artifacts/ref_time_df.csv', index=False)
    logging.info(f"Time reference generated: {len(ref_time)} samples.")
    df.to_csv('artifacts/pre_index_df.csv', index=False)
    df = df.set_index(index_cols)
    df.to_csv('artifacts/post_indexed_df.csv', index=False)
    logging.info(f"Data frame indexed with index columns: {index_cols}")
    logging.info(f"Index:\n{df.index}")
    logging.info(f"Uptime index values:\n{df.index.get_level_values('uptime')}")
    index_df = df.index.to_frame()
    index_df.to_csv('artifacts/orig_index.csv', index=False)
    # indexed_df.to_csv(os.path.join(artifact_path, 'indexed_df.csv'), index=True)
    # 2. Use a MultiIndex from_product to create the 'full' grid
    # This effectively does the "merge" for all 15,000 groups simultaneously
    full_index = pd.MultiIndex.from_product([
        df.index.levels[0], # site_names
        df.index.levels[1], # log_dates
        df.index.levels[2], # cellids
        df.index.levels[3], # ueids
        ref_time['uptime'].unique() # full uptime index
    ], names=['site_name', 'log_date', 'cellid', 'ueid', 'uptime'])
    df_padded = df.reindex(full_index, fill_value=0).sort_index()
    padded_idx = df_padded.index.to_frame()
    padded_idx.to_csv('artifacts/padded_idx.csv', index=False)
    logging.info(f"Padded sessions grid generated: {len(df_padded)} samples.")
    feature_array = df_padded[feature_cols].values.astype(np.float32)
    X = torch.tensor(np.array(feature_array)).reshape(-1, seq_len, len(feature_cols))
    logging.info(f"Created X tensor of shape: {X.shape}")
    if return_y:
        # Use the original df grouped by index to get one label per session
        # Ensure the order matches the unique combinations in your full_index
        grouped = df.groupby(level=[0, 1, 2, 3])
        
        # Extracting the first available label per group
        y_start_raw = grouped['issue_start'].first().values.astype(np.float32)
        y_rca_raw = grouped['rca_label'].first().values.astype(np.int64)

        y_start = torch.tensor(y_start_raw / (max_uptime + resolution), dtype=torch.float32)
        y_rca = torch.tensor(y_rca_raw, dtype=torch.long)
        logging.info(f"Created start tensor of length: {len(y_start)}")
        logging.info(f"Created RCA tensor of length: {len(y_rca)}")

        return X, y_start, y_rca
    else:
        return X
