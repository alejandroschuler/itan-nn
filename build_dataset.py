""" Reformat the raw ITAN dataset to digestible format
"""
import logging
import argparse
import os, shutil
from tqdm import tqdm
from pprint import pprint, pformat
from pathlib import Path

import json
import math
import numpy as np
import pandas as pd
# import feather
from scipy import signal
import peakutils
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)

import utils


def compress_raw_dataset(in_file, key, out_file=None, complevel=9, chunksize=10**6):
    """ Compress raw .tsv files to HDF5 table format
    """
    if out_file is None: out_file = "{}_{}.{}".format(os.path.splitext(in_file)[0], complevel, 'h5')
    logging.info("\n\nCompressing dataset from {} to {} at level {}".format(in_file, out_file, complevel))
    df_chunks = pd.read_csv(in_file, sep='\t', chunksize=chunksize)
    num_rows = utils.count_csv_rows(in_file, sep='\t')
    num_chunks = math.ceil(num_rows / chunksize)
    logging.info("Reading {} chunks of size {} from {} rows".format(num_chunks, chunksize, num_rows))
    start_flag = True
    with tqdm(total=num_chunks) as t:
        for chunk in df_chunks:
            if start_flag:
                start_flag = False
                append = False  # overwrite/restart table
                # logging.info("Read {} records from {}".format(len(df), in_file))
                logging.info("Columns:\n{}".format(pformat(chunk.columns)))
                logging.info("Data types:\n{}".format(pformat(chunk.dtypes)))
                logging.info("Examples:\n{}".format(pformat(chunk[:5])))
                # logging.info("Unique MRNs: {}".format(len(df['sim:MRN'].unique())))
            else:
                append = True # append chunk to end of HDF5 table

            chunk.to_hdf(out_file, key=key, append=append, mode='a', format='t', complib='bzip2', complevel=complevel)
            t.update()
    # df.to_feather(out_file)
    # feather.write_dataframe(df, out_file)

def build_dataset():
    """ Build the feature/label matrix for prediction problem, and split for cross-validation
    """
    logging.info("Building dataset")

def prepend_column_name(name, columns):
    """ Find sim/src:name and prepend appropriately to name
    """
    if "sim"+name in columns:
        return "sim"+name
    elif "src"+name in columns:
        return "src"+name
    else:
        logging.warning("{} column not found in dataframe".format(name))


def build_patient_split_dataset(in_file, out_file, chunksize=10**6, mrn_column='sim:PAT_MRN_ID'):
    logging.info("\n\nSplitting dataset from {} to {} wtih patient-level keys".format(in_file, out_file))
    # df_chunks = pd.read_hdf(in_file, chunksize=chunksize)
    # num_rows = len(df_chunks)  # Doesn't work with TableIterator
    df_chunks = pd.read_csv(in_file, sep='\t', chunksize=chunksize)
    num_rows = utils.count_csv_rows(in_file, sep='\t')

    num_chunks = math.ceil(num_rows / chunksize)
    logging.info("Reading {} chunks of size {} from {} rows".format(num_chunks, chunksize, num_rows))
    start_flag = True
    with tqdm(total=num_chunks, desc="Chunks") as t1:
        for ci, chunk in enumerate(df_chunks):
            if start_flag:
                start_flag = False
                append = False  # overwrite/restart table
                # logging.info("Read {} records from {}".format(len(df), in_file))
                logging.info("Columns:\n{}".format(pformat(chunk.columns)))
                logging.info("Data types:\n{}".format(pformat(chunk.dtypes)))
                # logging.info("Examples:\n{}".format(pformat(chunk[:5])))
                # logging.info("Unique MRNs: {}".format(len(df['sim:MRN'].unique())))
            else:
                append = True # append chunk to end of HDF5 table

            chunk_mrns = chunk[mrn_column].unique()
            logging.debug("{} MRNs in chunk {}".format(len(chunk_mrns), ci))
            with tqdm(total=len(chunk_mrns), desc="MRNs in Chunk") as t2:
                for mrn in chunk_mrns:
                    pt_chunk = chunk.loc[(chunk[mrn_column] == mrn), :]
                    logging.debug("Appending {} samples for mrn {}".format(len(pt_chunk), mrn))
                    pt_chunk.to_hdf(out_file, key="MRN_"+str(mrn), append=append, mode='a', format='t')#, complib='bzip2', complevel=complevel)
                    t2.update()
            t1.update()
            # break

def clean_data_dir(data_dir):
    logging.info("Removing all .h5 files from {}".format(data_dir))
    filelist = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    logging.info("{} files will be removed".format(len(filelist)))
    for f in filelist:
        os.remove(os.path.join(data_dir, f))


def split_encounter_files(
        cohort_file,
        hourly_file,
        out_dir,
        cohort_encounter_column="encounter_id",
        hourly_encounter_column="pat_enc_csn_id",
        chunksize=10**6,
        split_cohort=False):
    logging.info("Splitting dataset to individual files for each encounter ID to {}".format(out_dir))
    utils.ensure_directory(out_dir)
    clean_data_dir(out_dir)

    # Initialize all encounter files with cohort data
    if split_cohort:
        logging.info("Splitting cohort-level data from {} to individual encounter files".format(cohort_file))
        cohort_df = pd.read_csv(cohort_file, sep='\t')
        cohort_encounter_ids = sorted(cohort_df[cohort_encounter_column].unique())
        with tqdm(total=len(cohort_df)) as t:
            for ei, encounter_id in enumerate(cohort_encounter_ids):
                enc_chunk = cohort_df.loc[(cohort_df[cohort_encounter_column] == encounter_id), :]
                encounter_filepath = os.path.join(out_dir, "enc_{}.h5".format(encounter_id))
                enc_chunk.to_hdf(encounter_filepath, key="cohort", append=True, mode='a', format='t')
                t.update()
                # if ei >= 99: break

    logging.info("Splitting hourly-level data from {} to individual encounter files".format(hourly_file))
    hourly_df_chunks = pd.read_csv(hourly_file, sep='\t', chunksize=chunksize)
    num_rows = utils.count_csv_rows(hourly_file, sep='\t')

    num_chunks = math.ceil(num_rows / chunksize)
    logging.info("Reading {} chunks of size {} from {} rows".format(num_chunks, chunksize, num_rows))
    start_flag = True
    unknown_encounters = set()
    known_encounter_count = 0
    with tqdm(total=num_chunks, desc="Chunks") as t1:
        for ci, chunk in enumerate(hourly_df_chunks):
            if ci == 0:
                logging.info("Data types:\n{}".format(pformat(chunk.dtypes)))

            chunk_encounter_ids = sorted(chunk[hourly_encounter_column].unique())
            with tqdm(total=len(chunk_encounter_ids), desc="Encounter IDs in Chunk") as t2:
                for ei, encounter_id in enumerate(chunk_encounter_ids):
                    enc_chunk = chunk.loc[(chunk[hourly_encounter_column] == encounter_id), :]
                    encounter_filepath = os.path.join(out_dir, "itan_hourly_enc_{}.h5".format(encounter_id))
                    if not os.path.isfile(encounter_filepath):
                        logging.debug("Adding hourly data to encounter with unknown cohort data: {}".format(encounter_id))
                        unknown_encounters.add(encounter_id)
                    else:
                        known_encounter_count += 1
                    logging.debug("Appending {} samples for encounter ID {} to {}".format(len(enc_chunk), encounter_id, encounter_filepath))
                    enc_chunk.to_hdf(encounter_filepath, key="hourly", append=True, mode='a', format='t')#, complib='bzip2', complevel=complevel)
                    t2.update()
                    # if ei >= 99: break
            t1.update()
            # if ci >= 0: break

    logging.info("Added hourly data to {} encounter files with unknown IDs.  {} with known IDs".format(
        len(unknown_encounters), known_encounter_count))


def check_encounter_file(filepath):
    logging.info("Checking all contents of encounter file {}".format(filepath))
    try:
        cohort_df = pd.read_hdf(filepath, key="cohort").reset_index()
        logging.info("Found {} cohort samples:\n{}".format(len(cohort_df), cohort_df))
    except:
        logging.info("Could not find cohort data")

    try:
        hourly_df = pd.read_hdf(filepath, key="hourly").reset_index()
        logging.info("Found {} hourly samples:\n{}".format(len(hourly_df), hourly_df[:5]))
    except:
        logging.info("Could not find hourly data")


def extract_features_from_hourly(hourly_dir, out_file, features=[], feature_counts=[], save_period=1000):
    """ Extract a given number of specified features from hourly dataset for each encounter
    """
    logging.info("Extracting first {} samples of {} features for all encounters in {}".format(
        feature_counts, features, hourly_dir))
    assert len(features) == len(feature_counts), "Need a count for each feature"

    # Check all encounter hourly files
    hourly_files = [f for f in os.listdir(hourly_dir) if f.endswith('h5')]
    encounter_ids = [np.int64(f.split('.')[0].split('_')[-1]) for f in hourly_files]
    logging.info("Found {} hourly encounter IDs".format(len(encounter_ids)))

    # Initialize feature matrix
    feature_columns = ["{}_{}".format(features[fi], ri) for fi in range(len(features)) for ri in range(feature_counts[fi])]
    feature_matrix = pd.DataFrame(index=encounter_ids, columns=feature_columns)
    # logging.info("Feature Matrix: {}\n{}".format(feature_matrix.shape, feature_matrix[:5]))

    with tqdm(total=len(encounter_ids)) as t:
        for ei, encounter_id in enumerate(encounter_ids):
            hourly_df = pd.read_hdf(os.path.join(hourly_dir, hourly_files[ei]), key="hourly").reset_index(drop=True)
            hourly_df.columns = [c.split(':')[1] for c in hourly_df.columns] # clean sim/src headers for fuzzed dataset
            logging.debug("Hourly DataFrame for encounter {}:\n{}".format(encounter_id, hourly_df[:3]))
            for fi, feature in enumerate(features):
                for ri in range(min(len(hourly_df), feature_counts[fi])):
                    column = "{}_{}".format(feature, ri)
                    feature_matrix.loc[encounter_id, column] = hourly_df.loc[ri, feature]

            if (ei > 0) and ((ei % save_period) == 0):
                logging.debug("Saving at {} encounters".format(ei))
                feature_matrix.to_hdf(out_file, key="hourly")
            t.update()
            # break

    logging.info("Feature Matrix: {}\n{}".format(feature_matrix.shape, feature_matrix[:5]))
    feature_matrix.to_hdf(out_file, key="hourly")


if __name__ == '__main__':
    # Set the logger
    utils.set_logger('./log/build_dataset.log', logging.INFO) #  DEBUG  INFO

    # compress_raw_dataset("../Data/itan_cohort_v.tsv", key='cohort')#, out_file='../Data/itan_fuzzed_v.h5')
    # compress_raw_dataset("../Data/itan_hourly_bed_unit_v.tsv", key='hourly_bed_unit')#, out_file='../Data/itan_fuzzed_v.h5')
    # compress_raw_dataset("../Data/itan_patient_hourly_v.tsv", key='patient_hourly')#, out_file='../Data/itan_fuzzed_v.h5')

    # build_patient_split_dataset("../Data/itan_patient_hourly_v.tsv", "../Data/itan_patient_hourly_v_patientkeyed.h5")

    itan_data = f"{Path.home()}/data/itan"
    # Split the dataset into individual .h5 files for each patient encounter
    split_encounter_files(cohort_file = f"{itan_data}/cohort.tsv",
                          hourly_file = f"{itan_data}/hourly.tsv",
                          out_dir = f"{itan_data}/encounter_splits/",
                          chunksize=10**5)


    # check_encounter_file(os.path.join("../Data/itan_hourly_encounter_splits/", "itan_hourly_enc_330000995550.h5"))
    # check_encounter_file(os.path.join("../Data/itan_hourly_encounter_splits/", "itan_hourly_enc_330000125762.h5"))

    # extract_features_from_hourly(
    #     hourly_dir = "../Data/itan_hourly_encounter_splits/",
    #     out_file = "../Data/itan_hourly_extracted.h5",
    #     features=["COPS2", "LAPS2", "LOCATION"],
    #     feature_counts=[6, 6, 1])

