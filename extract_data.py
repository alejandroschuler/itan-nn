from env import rdb, data_dir

from pathlib import Path
from tqdm import tqdm
import pandas as pd

def fetchall_df(result_proxy):
    """
    Works like sqlalchemy.engine.result.ResultProxy.fetchall(), 
    but shows progress reading in the data and returns a pandas 
    dataframe with correct column names instead of a list of tuples 
    (rows) with no column names
    """
#     result = result_proxy.fetchall(keep_col_names=T) ???
    result = [row for row in tqdm(result_proxy)]
    return pd.DataFrame(result, columns=result[0].keys())

hourly_q = """
    select * 
    from doretl.ITAN_PATIENT_HOURLY_V
"""
cohort_q = """
    select * 
    from doretl.ITAN_COHORT_V
"""
queries = (hourly_q, cohort_q)
filenames = ("hourly.tsv", "cohort.tsv")

for query, filename in zip(queries, filenames):
    result = rdb.execute(query)
    fetchall_df(result).to_csv(data_dir/filename, sep="\t")
