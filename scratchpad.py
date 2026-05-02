import os
import pandas as pd
import numpy as np
from src.components.ingestion import Ingestion

if __name__=="__main__":
    max_uptime = 28770
    resolution = 30
    artifact_path = 'artifacts'
    ingestion_obj = Ingestion(artifact_path)
    data_filepath = ingestion_obj.ingest_data('1970-01-01', 32)
    df = pd.read_csv(data_filepath)
    ref_time = pd.DataFrame(data=np.arange(0, max_uptime+resolution, resolution), columns=['uptime'])
    indexed_df = df.set_index(['site_name', 'log_date', 'cellid', 'ueid', 'uptime'])
    # indexed_df.to_csv(os.path.join(artifact_path, 'indexed_df.csv'), index=True)
    # 2. Use a MultiIndex from_product to create the 'full' grid
    # This effectively does the "merge" for all 15,000 groups simultaneously
    full_index = pd.MultiIndex.from_product([
        indexed_df.index.levels[0], # site_names
        indexed_df.index.levels[1], # log_dates
        indexed_df.index.levels[2], # cellids
        indexed_df.index.levels[3], # ueids
        ref_time['uptime'].unique()
    ], names=['site_name', 'log_date', 'cellid', 'ueid', 'uptime'])
    df_padded = indexed_df.reindex(full_index, fill_value=0)
    df_padded.to_csv(os.path.join(artifact_path, 'padded_df.csv'), index=True)