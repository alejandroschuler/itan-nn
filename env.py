import json
from pathlib import Path
import sqlalchemy as db

# replace this with keyring once Lawrence gets that working
# credentials.json file looks like:
# {
#     "username": "c242587",
#     "password": "MY_PASSWORD",
#     "host": "dor-m9-rdb.nndc.kp.org"
# }
with open(Path.home() / 'credentials.json') as f:
    data = json.load(f)
    username = data['username']
    password = data['password']
    host = data['host']

# %load_ext sql
# %sql oracle+cx_oracle://$username:$password@$host:1521/?service_name=dororat3
rdb_engine = db.create_engine(f'oracle+cx_oracle://{username}:{password}@{host}:1521/?service_name=dororat3')
rdb = rdb_engine.connect()

from pathlib import Path

data_dir = Path.home() / 'data' / 'itan'
code_dir = Path.home() / 'code' / 'itan-nn'