import os
from dotenv import load_dotenv
from databricks.sql import connect
from typing import List
from logger.logger import logging
from tqdm.auto import tqdm
import torch
import pandas as pd

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

def fetch_data(log_date:str, site_name:str):
    try:
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
        with tqdm(total=num_rows, desc=f"Fetching data for {log_date, site_name}...", unit="row",disable=True) as pbar:
            with connect(
                server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                http_path=os.getenv("DATABRICKS_HTTP_PATH"),
                access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    while True:
                        row = cursor.fetchmany_arrow(batch_size)
                        if row:
                            row_df = row.to_pandas()
                            df = pd.concat([df, row_df])
                            pbar.update(batch_size)
                        else:
                            break

        return df
    except Exception as e:
        logging.info(f"Could not fetch data for {log_date, site_name}: {e}")

def accuracy_fn(y_logits: torch.Tensor, y_true: torch.Tensor)->float:
    y_pred = torch.argmax(y_logits, dim=1)*30
    y_true = y_true * 30

    return torch.eq(y_pred, y_true).sum().item()/len(y_true)*100

def mae_eval(y_pred:torch.Tensor, y_true:torch.Tensor)->float:
    return torch.abs(y_pred - y_true).sum().item()/len(y_true)

def get_inverse_class_weights(y_label:torch.Tensor, num_classes:int):
    class_weights = []
    for i in range(num_classes):
        class_samples = (y_label == i).sum()
        class_weights.append(len(y_label)/num_classes/class_samples)
    return torch.Tensor(class_weights)

def eval_loss(y_pred, y_true, mask, weights):

    y_start_pred = y_pred[0]
    y_rca_label_logits = y_pred[1]

    y_start_true = y_true[0]
    y_rca_label_true = y_true[1]

    class_weigths = get_inverse_class_weights(y_rca_label_true, 4)

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

    loss = eval_loss(y_pred, y_true, mask, (100, 1))
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

      loss += eval_loss(y_pred, y_true, mask, (1, 1))
      start_mae += mae_eval((y_start_pred*mask*960).to(torch.int)*30, (y_start_test*mask*960).to(torch.int)*30)
      rca_acc += accuracy_fn(y_rca_label_logits, y_rca_label_test)

  return loss.item()/len(test_loader), start_mae/len(test_loader), rca_acc/len(test_loader)