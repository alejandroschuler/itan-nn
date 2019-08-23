import pandas as pd
from env import data_dir
from pathlib import Path
from tqdm import tqdm

def tsv_to_h5(file_path, key):
    pd.read_csv(
        file_path,
        sep='\t'
    ).to_hdf(
        file_path.with_suffix('.h5'), 
        key=key, 
        mode='a', format='t'
    )

# not sure what the "key" arg is for so I set it as Stanford people did
# tsv_to_h5(data_dir/"cohort.tsv", key="cohort")

enc_files = (data_dir/'encounter_splits').glob('*.tab')
for f in tqdm(enc_files):
    tsv_to_h5(f, key="hourly")
    f.unlink()
