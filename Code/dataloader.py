""" Data loader to dynamically allocate batches and pass to Skorch
    Primarily utilizes pandas and numpy

Author:     Daniel Roy Miller
Created:    5/7/2019
"""
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import logging
import argparse
import os
from pprint import pprint, pformat

# Basis Expansion (external package)
from basis_expansions import (Binner, Polynomial, 
                              LinearSpline, CubicSpline,
                              NaturalCubicSpline)
from dftransformers import ColumnSelector, FeatureUnion, Intercept, MapFeature

# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Deep Learning
import torch
from torch import nn
import torch.nn.functional as F
import skorch
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

# Local files
import utils


class ITANStrainDataset(skorch.dataset.Dataset):
    def __init__(self, sample_ids, params, mode='train', trained_params=None, show_progress_bar=True):
        """ Construct ITAN Dataset for dynamic data loading with Skorch
        
        Args:
            sample_ids: (pandas.Series or list)  Sample (encounter) ids for indexing cohort data and hourly files
            params: (dict)  Network and training parameters
            mode: (string)  train/eval mode.  Controls normalization of hourly data.  i.e. fit on train data, apply to test data
            trained_params: (dict) pre-computed mean and standard deviation of hourly features for normalization.
            show_progress_bar: (bool) whether to show tqdm progress bars on initialization and preprocessing steps
        """
        self.show_progress_bar = show_progress_bar
        self.params = params
        self.sample_ids = list(set(sample_ids))  # drop duplicates from [list]
        logging.debug("Sample IDs:  {}\n{}".format(len(self.sample_ids), self.sample_ids[:5]))
        
        # Keep all cohort data in memory
        if self.params.cohort_features:
            self.cohort_df = self.load_cohort_features()
            self.cohort_features = [c for c in self.cohort_df.columns if c != self.params.label]
            
            # Drop samples that don't have cohort data
            self.sample_ids = self.cohort_df.index
            
        # Split features and label/target
        # Raw length of stay is in days
        self.length_of_stay = self.cohort_df.loc[:, [params.label]].values.astype(np.float32)

        self.cohort_df.drop([params.label], axis=1, inplace=True)
        
        if self.params.normalize_cohort:
            self.cohort_df = self.normalize(self.cohort_df)
            
        # Keep all hourly data in memory
        if self.params.hourly_features:
            self.hourly_features = self.params.hourly_features
            self.hourly_dataframes, self.global_time_index, self.all_hourly_lengths = self.load_hourly_features(self.sample_ids)
            
            if self.params.normalize_hourly:
                if mode == 'train':
                    hourly_feature_means, hourly_feature_vars = self.compute_hourly_normalization_coefficients(self.hourly_dataframes, self.sample_ids)
                else:
                    hourly_feature_means = trained_params["hourly_feature_means"]
                    hourly_feature_vars = trained_params["hourly_feature_vars"]
                self.hourly_dataframes = self.normalize_hourly_features(self.hourly_dataframes, self.sample_ids, hourly_feature_means, hourly_feature_vars)

        # Prebuild all patient locations
        self.hourly_patient_locations, self.all_locations = self.prebuild_patient_locations(self.sample_ids, self.hourly_dataframes, self.global_time_index)
        self.all_locations_index_dict = {loc: li for li, loc in enumerate(self.params.all_locations)}
        
        # Prebuild all patient associations to index at batch-time
        self.patient_associations = self.prebuild_patient_associations(self.sample_ids,
                                                                       self.hourly_patient_locations,
                                                                       history_hours = self.params.association_history)

        # Gather trained parameters for re-use in future datasets
        self.trained_params = {
            "hourly_feature_means": hourly_feature_means,
            "hourly_feature_vars": hourly_feature_vars
        }

        # Initialize targets which are filled through __getitem__, all valid after 1 epoch
        # Note:  is reshaped in valid length restriction
        if self.params.predict_after_hours > 0:
            self.target = np.zeros(shape=self.length_of_stay.shape, dtype=np.float32)
        else:
            self.target = np.zeros(shape=(self.length_of_stay.shape[0], self.params.pad_length), dtype=np.float32)
        if self.params.output_type == "classification":
            logging.debug("Expanding initialized targets for classification from shape: {}".format(self.target.shape))
            logging.debug("Expanded targets: {}".format(np.expand_dims(self.target, len(self.target.shape)).shape))
            logging.debug("repeat list: {}".format([1]*len(self.target.shape)+[2]))
            self.target = np.tile(np.expand_dims(self.target, len(self.target.shape)), [1]*len(self.target.shape)+[2])
            logging.debug("New initialized targets shape: {}".format(self.target.shape))

        # By default, drop any samples which have less valid data than the predict_after_hours
        self.restrict_to_valid_length_samples(self.params.predict_after_hours)

        
    def normalize(self, df):
        """ Normalize all columns in a dataframe (cohort_df) to mean 0 and std 1
            Based on sklearn.preprocessing.StandardScaler
        """
        logging.debug("Normalizing features")
        scaler = StandardScaler()
        x = df.values #returns a numpy array
        x_scaled = scaler.fit_transform(x)
        df[:] = x_scaled
        return df
            
    def load_cohort_features(self, force_categorical_features=["HCUPSG"]):
        """ Load all cohort-level features from file and preprocess

        Args:
            force_categorical_features: (list) Which features to force to categorical (rather than just detecting data type)
                                               Relevant for numerically coded classes (i.e. HCUPSG)
        Return:
            cohort_df:  (pandas.DataFrame)  All cohort-level data indexed by encounter ID
        """
        logging.info("Pre-loading all cohort features")
        cohort_df = pd.read_hdf(os.path.join(self.params.project_dir, self.params.data_dir, self.params.cohort_file),
                                start=0, 
                                stop=-1) # note: loading all features and dropping after cleaning sim/src
        cohort_df.columns = [c.split(':')[1] for c in cohort_df.columns] # clean sim/src headers for fuzzed dataset
        cohort_df.set_index("ENCOUNTER_ID", inplace=True)
        cohort_df = cohort_df.loc[:, [self.params.label]+self.params.cohort_features]
        logging.debug("Loaded {} samples from cohort-level data".format(cohort_df.shape))
        
        # Drop rows with duplicate Encounter IDs
        cohort_df = cohort_df[~cohort_df.index.duplicated(keep='first')]
        logging.debug("{} samples after dropping duplicate encounter IDs".format(cohort_df.shape))
        
        # Select out the rows for encounter IDs in the current CV split
        cohort_df = cohort_df.loc[self.sample_ids, :]
        logging.info("Loaded cohort data:  {}\n{}".format(cohort_df.shape, cohort_df[:5]))
        logging.debug("Cohort datatypes:\n{}".format(cohort_df.dtypes))
        
        # Drop rows with NaN
        cohort_df.dropna(axis='index', how='any', inplace=True)
        logging.debug("{} samples after dropping rows with missing values".format(len(cohort_df)))
        
        # Expand categorical features with dummy indicators
        # Start by attempting to detect by datatype (easy if strings)
        categorical_features = [c for ci, c in enumerate(cohort_df.columns) if cohort_df.dtypes[ci]==object]
        # Add explicitly defined categoricals (e.g. if categories are defined by numerical codes)
        categorical_features = list(set(categorical_features +
                                        [f for f in force_categorical_features if f in cohort_df.columns]))
        if categorical_features:
            logging.debug("Expanding categorical variables to dummy indicators:  {}".format(categorical_features))
            cohort_df = pd.get_dummies(cohort_df, columns=categorical_features).astype(np.float32)
            logging.debug("Expanded/binarized data:\n{}".format(cohort_df[:5]))
        else:
            logging.debug("No categorical variables to expand")
        
        # Return the expanded cohort data, with labels (targets, Y) included
        return cohort_df
    
    def load_hourly_features(self, sample_ids, location_column="LOCATION", index_column="LAPS2_TS", drop_rows=True):
        """ Load hourly data for all patients listed in self.sample_ids
        
        Args:
            sample_ids: (list)  Which sample IDs to load data for

        Returns:
            hourly_dataframes: (dictionary of DataFrame, keyed by sample_id)
            global_time_index:  pd.DateTimeIndex?  hourly indices over all give sample IDs
        """
        logging.info("Pre-loading all hourly features")
        hourly_dataframes = {}
        all_hourly_lengths = pd.Series(index=sample_ids)
        with tqdm(total=len(sample_ids), desc="Sample IDs", disable=~self.show_progress_bar) as ts:
            for sample_id in sample_ids:
                filepath = os.path.join(self.params.project_dir,
                                        self.params.sample_dir,
                                        "itan_hourly_enc_{}.h5".format(sample_id))

                # Load patient hourly data
                sample_hourly_df = pd.read_hdf(filepath, key="hourly").reset_index()
                sample_hourly_df.drop('index', axis=1, inplace=True)
                sample_hourly_df.columns = [c.split(':')[1] for c in sample_hourly_df.columns] # clean sim/src headers for fuzzed dataset

                # Set index to start of each hour (rounded to :00) and drop unused hourly features
                sample_hourly_df[index_column] =  pd.to_datetime(sample_hourly_df[index_column]).dt.round('1H')  
                sample_hourly_df.set_index(index_column, inplace=True)
                sample_hourly_df = sample_hourly_df.loc[:, [location_column]+self.params.hourly_features]
                

                # Drop all rows with any missing data
                # sample_hourly_df.dropna(inplace=True)
                sample_hourly_df.fillna(0, inplace=True)
                logging.debug("Sample {} hourly dataframe:\n{}".format(sample_id, sample_hourly_df[:5]))

                # Compute the quantity of valid data (number of hours) in the current sample
                sample_pad = min(self.params.pad_length, len(sample_hourly_df))
                all_hourly_lengths[sample_id] = sample_pad

                # Retain all hourly data in CPU memory
                # Will be drawn from to construct training batches, which are moved to GPU
                hourly_dataframes[sample_id] = sample_hourly_df
                ts.update()
            
        # Check time frames and build global time index
        start_stop_times = pd.DataFrame(data=[[sample_hourly_df.index.min(), sample_hourly_df.index.max()]
                                              for sample_id, sample_hourly_df in hourly_dataframes.items()],
                                        columns=["Start", "Stop"])
        
        logging.debug("Start/stop times for each sample ID:\n{}".format(start_stop_times))
        global_time_index = pd.date_range(start=start_stop_times["Start"].min(),
                                            end=start_stop_times["Stop"].max(),
                                            freq="1H")
        logging.info("Global time indices over {} hours:  type {}\n{}".format(len(global_time_index), type(global_time_index), global_time_index[:5]))

        # Drop unnecessary rows to save memory
        keep_count = 0
        row_count = 0
        if drop_rows:
            logging.info("Dropping rows not in global index from hourly data to save memory")
            for sample_id, sample_hourly_df in hourly_dataframes.items():
                row_count += len(sample_hourly_df)
                valid_time_index = sample_hourly_df.index.intersection(global_time_index)
                hourly_dataframes[sample_id] = sample_hourly_df.loc[valid_time_index, :]
                keep_count = len(sample_hourly_df)
            logging.info("Reduced hourly data size to {:.2f} percent of original".format(100. * keep_count / row_count))
        return hourly_dataframes, global_time_index, all_hourly_lengths
        
    def compute_hourly_normalization_coefficients(self, hourly_dataframes, sample_ids):
        """ Compute the mean and std of all hourly features for normalization.
            Uses rolling/pooled computation to allow loading all hourly data 1 patient at a time.

        Args:
            hourly_dataframes: (dict of DataFrame) pre-loaded raw hourly data, keyed by patient encounter ID
            sample_ids: (list) which encounter IDs to compute the coefficients over

        Returns:
            overall_feature_means: mean for each hourly feature over all sample_ids
            overall_feature_vars:  variance for each hourly feature over all sample_ids
        """
        logging.info("Computing normalization means and variances for all hourly features")
        hourly_feature_means = pd.DataFrame(index=sample_ids,
                                            columns=self.params.hourly_features)
        hourly_feature_vars = pd.DataFrame(index=sample_ids,
                                           columns=self.params.hourly_features)
        hourly_feature_counts = pd.Series(index=sample_ids)

        logging.info("Computing mean and variance over all sample IDs")
        with tqdm(total=len(sample_ids), desc="Sample IDs", disable=~self.show_progress_bar) as ts:
            for sample_id in sample_ids:
                hourly_feature_means.loc[sample_id, self.params.hourly_features] = hourly_dataframes[sample_id].loc[:, self.params.hourly_features].mean(axis=0)
                hourly_feature_counts[sample_id] = len(hourly_dataframes[sample_id])
                hourly_feature_vars.loc[sample_id, self.params.hourly_features] = hourly_dataframes[sample_id].loc[:, self.params.hourly_features].var(axis=0)
                ts.update()
        logging.debug("Feature means for all sample IDs  (shape {}):\n{}".format(hourly_feature_means.shape, hourly_feature_means))
        logging.debug("Feature variances for all sample IDs  (shape {}):\n{}".format(hourly_feature_vars.shape, hourly_feature_vars))

        # Compute pooled mean
        logging.debug("sample counts:\n{}".format(hourly_feature_counts))
        logging.debug("weighted means:\n{}".format(hourly_feature_means.multiply(hourly_feature_counts, axis="index")))
        logging.debug("total sums:\n{}".format(hourly_feature_means.multiply(hourly_feature_counts, axis="index").sum(axis=0)))
        logging.debug("total count:\n{}".format(hourly_feature_counts.sum()))
        overall_feature_means = hourly_feature_means.multiply(hourly_feature_counts, axis="index").sum(axis=0) / hourly_feature_counts.sum()
        logging.info("Overall feature means:\n{}".format(overall_feature_means))
        
        # Compute pooled variance
        # https://en.wikipedia.org/wiki/Pooled_variance
        overall_feature_vars = hourly_feature_vars.multiply((hourly_feature_counts-1), axis="index").sum(axis=0) / (hourly_feature_counts - 1).sum()
        logging.info("Overall feature variances:\n{}".format(overall_feature_vars))
        return overall_feature_means, overall_feature_vars

    def normalize_hourly_features(self, hourly_dataframes, sample_ids, overall_feature_means, overall_feature_vars):
        """ Use the precomputed mean/variance to normalize all hourly data for all features
        
        Args:
            hourly_dataframes: (dict of DataFrame) pre-loaded raw hourly data, keyed by patient encounter ID
            sample_ids: (list) which encounter IDs to apply the normalization to
            overall_feature_means: mean for each hourly feature over all sample_ids
            overall_feature_vars:  variance for each hourly feature over all sample_ids

        Returns:
            hourly_dataframes: (dict of DataFrame) similar to input, but each feature normalized to mean 0 and std 1
        """
        logging.info("Normalizing by overall mean/var for all sample IDs")
        with tqdm(total=len(sample_ids), desc="Sample IDs", disable=~self.show_progress_bar) as ts:
            for sample_id in sample_ids:
                # logging.info("raw hourly features (shape {}):\n{}".format(hourly_dataframes[sample_id].loc[:, self.params.hourly_features].shape, hourly_dataframes[sample_id].loc[:, self.params.hourly_features]))
                # logging.info("centered:\n{}".format(hourly_dataframes[sample_id].loc[:, self.params.hourly_features].sub(overall_feature_means, axis='columns')))
                # logging.info("standard deviations:\n{}".format(overall_feature_vars.pow(.5)))
                hourly_dataframes[sample_id].loc[:, self.params.hourly_features] = hourly_dataframes[sample_id].loc[:, self.params.hourly_features].sub(overall_feature_means, axis='columns').div(overall_feature_vars.pow(.5), axis="columns")
                # logging.info("Normalized hourly features for Sample ID {}:\n{}".format(sample_id, hourly_dataframes[sample_id]))
                ts.update()
        logging.info("Hourly feature normalization complete!")
        return hourly_dataframes

    def prebuild_patient_locations(self, sample_ids, hourly_dataframes, global_time_index):
        """ Build matrix of patient locations at each time
            Note:  assumes single facility (FAC_ID not used)
        
        Args:
            sample_ids: (list) which encounter IDs to prebuild locations for
            hourly_dataframes: (dict of DataFrame) pre-loaded raw hourly data, keyed by patient encounter ID
            global_time_index: (Pandas DateTime Index) index for all location data

        Returns:
            hourly_patient_locations: (pd.DataFrame)  shape=(global_time_max x num_patients)
            all_locations: (list) string values of all unique unit locations
        """
        logging.info("Pre-building all patient locations in global time index")
        
        hourly_patient_locations = pd.DataFrame(columns=sample_ids, index=global_time_index)
        with tqdm(total=len(sample_ids), desc="Sample IDs", disable=~self.show_progress_bar) as ts:
            for sample_id in sample_ids:
                sample_times = hourly_dataframes[sample_id].index
                hourly_patient_locations.loc[sample_times, sample_id] = hourly_dataframes[sample_id]["LOCATION"]
                ts.update()
        logging.debug("Built patient locations: {}:\n{}".format(hourly_patient_locations.shape, hourly_patient_locations[:3]))
        
        # Build list of all possible locations.  Location matrix values will be indices into this list
        all_locations = hourly_patient_locations.unstack().dropna().unique()
        logging.info("All possible locations:  {}".format(all_locations))
        
        return hourly_patient_locations, all_locations
    
    def prebuild_patient_associations(self, sample_ids, hourly_patient_locations, history_hours=0):
        """ Prebuild array for which patients are associated with which.
            Based on simultaneous location in same facility/unit, or prior occupation within a fixed time window
        
        Params:
            sample_ids:  (list)  which patients for which to find associated other patients (e.g. the batch samples)
            hourly_patient_locations:  (DataFrame)  the patient locations for all samples
            history_hours:  (int) number of hours to lookback when deciding if patients are associated
                            default 0 to require simultaneous occupancy of same unit
        
        Returns:
            associations:  (DataFrame of bool)  indicators for which patients are associated with the given sample patients
        """
        logging.info("Pre-building all patient associations")
        # all_sample_ids = hourly_patient_locations.columns
        associations = pd.DataFrame(0, dtype=np.bool, index=sample_ids, columns=sample_ids)
        
        with tqdm(total=len(sample_ids), desc="Sample IDs", disable=~self.show_progress_bar) as ts:
            for sample_id in sample_ids:
                # Create location history matching array
                # Note:  len(sample_ids) <= len(all_sample_ids)

                # Check which patients were in a matching location at some point in the history limit


                # TODO:  Haven't implemented historical occupation yet
                if (history_hours != 0): raise NotImplementedError

                all_locations = hourly_patient_locations.values
                logging.debug("All locations:  {}\n{}".format(hourly_patient_locations.shape, hourly_patient_locations[:3]))
                sample_locations = np.expand_dims(hourly_patient_locations.loc[:, sample_id].values, axis=1)
                logging.debug("\nSample {} Locations:  {}\n{}".format(sample_id, sample_locations.shape, sample_locations[:3]))
                matched_locations = (hourly_patient_locations.values == sample_locations)
                # matched_locations = np.equal(hourly_patient_locations.values, sample_locations) # == will reduce on singletons
                # logging.debug("\nMatched Location Indicators:  {}\n{}".format(matched_locations.shape, matched_locations[:3]))
                associated_patient_mask = np.any(matched_locations, axis=0)
                # logging.debug("\nMatched Location at any Time:  {}\n{}".format(associated_patient_mask.shape, associated_patient_mask[:10]))
                
                # Mask out which patients simultaneously occupy a unit with the current sample at any point in time
#                 associated_patient_mask = (hourly_patient_locations == hourly_patient_locations[sample_id]).any(axis='index')
                associations.loc[sample_id, sample_ids[associated_patient_mask]] = True
                ts.update()
                
        logging.debug("Patient associations: {}\n{}".format(associations.shape, associations[:3]))
        logging.info("Finished building patient associations:  shape {}".format(associations.shape))
        association_counts = associations.sum(axis=1)
        logging.info("Average of {}+/-{:.2f} associations per sample".format(association_counts.mean(), association_counts.std()))
        return associations
        
    def get_associated_patients_sample(self, sample_id, associations, hourly_patient_locations, all_locations):
        """ Get the sample IDs for all patients associated with sample_id
            Uses the prebuilt associations matrix
            
        Args:
            sample_id:  (string) the patient encounter ID to construct the sample for
            associations:  (DataFrame of bool)  indicators for which patients are associated with the given sample patients

        Returns:
            associated_patients:  (list) sample IDs associated with input sample ID
            hourly_patient_locations: (pd.DataFrame)  shape=(global_time_max x num_patients)
            all_locations: (list) string values of all unique unit locations
        """
        # Select out a single sample patient's associations from prebuilt DataFrame
        # all_sample_ids = self.restricted_sample_ids#associations.columns
        # Only allow association among the restricted sample IDs (those with sufficient data)
        associated_mask = associations.loc[sample_id, self.restricted_sample_ids].values
        logging.debug("Associated patient mask:  {}\n{}".format(associated_mask.shape, associated_mask[:10]))
        associated_patients = self.restricted_sample_ids[associated_mask]  # Create list of associated SampleIDs
        logging.debug("Associated Patients:  {}\n{}".format(associated_patients.shape, associated_patients[:10]))
        
        # Randomly select a fixed number of associated patients
        if len(associated_patients) > self.params.max_associated_patients:
            logging.debug("Found {} associated patients, greater than maximum of {}".format(len(associated_patients), self.params.max_associated_patients))
            associated_patients_sample_index = np.random.choice(a=len(associated_patients),
                                                                size=self.params.max_associated_patients,
                                                                replace=False)
            associated_patients = associated_patients[associated_patients_sample_index]
        # Select exactly N-1 associated patients who are not the sample patient
        associated_patients = [p for p in associated_patients if p != sample_id][:(self.params.max_associated_patients-1)]
        # Re-order so first sample ID in list is always the input sample ID
        associated_patients = [sample_id] + associated_patients
        logging.debug("Re-ordered Patients:  {}\n{}".format(len(associated_patients), associated_patients[:10]))
    
        # Create new time index for associated patients
        # Go backwards a fixed length from the latest timestamp for the current patient (sample_id)
        latest_time = hourly_patient_locations.loc[:, sample_id].dropna(how="all").index.max()
        logging.debug("Found latest time for sample {} as {}".format(sample_id, latest_time))
#         associated_patients_shared_time_index = self.global_time_index[self.global_time_index <= latest_time][-self.params.associated_time_max:]
        associated_patients_shared_time_index = pd.date_range(start=latest_time - pd.Timedelta(hours=self.params.associated_time_max-1),
                                                              end=latest_time,
                                                              freq="1H")  # note: this can go before the global time index for fixed-shape output
        logging.debug("Created new time index to share over associated patients:  {}\nfrom {} to {}\n{}".format(
            associated_patients_shared_time_index.shape, associated_patients_shared_time_index.min(), associated_patients_shared_time_index.max(), associated_patients_shared_time_index[:5]))
        
        # Create mask to scatter associate patient data over the new time index
        # [max_associations x assoc_time_max], X[i,j]=1 where patient i has data at time j
        # Will only send uint8 values to PyTorch
        # Assumes not throwing away any hourly data inside the shared time index
        # Assumes the associated patient hourly dataframe will start from the beginning of the shared index
        associated_patient_time_scatter_mask = pd.DataFrame(0, index=associated_patients_shared_time_index, columns=associated_patients)
        for asi, associated_patient in enumerate(associated_patients):
            # Select all time indices where the associated patient has hourly data
            associated_patient_index = hourly_patient_locations.loc[:, associated_patient].dropna(how='all').index
            # Mask out only those time indices that overlap the shared time index
            masked_associated_patient_index = associated_patient_index[associated_patient_index.isin(associated_patients_shared_time_index)]
            # Fill out the mask
            associated_patient_time_scatter_mask.loc[masked_associated_patient_index, associated_patient] = 1
        logging.debug("Scatter mask for associated patient's into shared time index:  {}\n{}".format(associated_patient_time_scatter_mask.shape, associated_patient_time_scatter_mask[:10]))
        
        # Create index to gather associated patients at each time by unit location
        # [num_locations x max_occupancy x max_time] must be broadcastable with the time-scattered patient hourly embeddings from LSTM
        # index[loc, pt, t] = index of pt into associated patients for all patients 'pt' occupying location 'loc' at time 't'
        # default to 0:  but junk/replicated data will be discarded later based on hourly_occupancies
        associated_patient_location_gather_index = np.zeros(shape=(len(all_locations), self.params.max_occupancy, self.params.associated_time_max), dtype=np.int64)
        associated_hourly_occupancies = pd.DataFrame(1, index=associated_patients_shared_time_index, columns=all_locations)
        first_global_time = hourly_patient_locations.index.min()
        for li, location in enumerate(all_locations):
            # only insert occupancies when data is available, prevent deprecation error
            valid_shared_time_index = associated_patients_shared_time_index.intersection(hourly_patient_locations.index)
                
            # Number of associated patients occupying each location at each hour in the shared time index
            associated_hourly_occupancies.loc[valid_shared_time_index, location] = (hourly_patient_locations.loc[valid_shared_time_index,
                                                                                                                 associated_patients] == location).sum(axis=1)
            # Clip to maximum occupancy, and minimum of 1 to allow valid indexing (index=occ-1)
            # If unit is empty should be masked by time, since output is only on sample times
            associated_hourly_occupancies[location].clip(lower=1, upper=self.params.max_occupancy, inplace=True)
            
            # Construct the gather index for matching unit-hourly strain data to patient hourly locations
            for ti, t in enumerate(associated_patients_shared_time_index):
                if t < first_global_time: continue # recall that shared time index may precede global index
                occupants = np.nonzero((hourly_patient_locations.loc[t, associated_patients] == location).values)[0] # index into nonzero() tuple since arg is 1D
#                 logging.debug("Occupants of {} at time {}:\n{}".format(location, t, occupants))
                if len(occupants) > self.params.max_occupancy:
                    associated_patient_location_gather_index[li, :, ti] = np.random.choice(occupants, size=self.params.max_occupancy, replace=False)
                else:
                    associated_patient_location_gather_index[li, :len(occupants), ti] = occupants
        logging.debug("Gather index over associated patients for hourly locations:  {}\n{}".format(associated_patient_location_gather_index.shape, associated_patient_location_gather_index[:1, :5, :5]))    
        logging.debug("Hourly occupancies over all units (count of associated patients):  {}\n{}".format(associated_hourly_occupancies.shape, associated_hourly_occupancies[:10]))
        
        return associated_patients, associated_patients_shared_time_index, associated_patient_time_scatter_mask, associated_patient_location_gather_index, associated_hourly_occupancies
    
    def restrict_to_valid_length_samples(self, required_length):
        """ Restrict which sample IDs to make predictions on
            Note:  can still use other samples as associated patients
        
        Args:
            required_length: (int) minimum number of hours to require

        Modifies:
            self.restricted_sample_ids
            self.length_of_stay
            self.target
            self.all_hourly_lengths
            self.patient_associations
        """

        if required_length > 0:
            self.restricted_sample_ids = list(self.all_hourly_lengths.loc[self.all_hourly_lengths >= required_length].index)
            self.length_of_stay = self.length_of_stay[(self.all_hourly_lengths >= required_length)]
            self.target = self.target[(self.all_hourly_lengths >= required_length)]
            self.all_hourly_lengths = self.all_hourly_lengths.loc[self.all_hourly_lengths >= required_length]
            # Restrict the rows (which patient to find associations for),
            # as well as the columns (which patients are allowed to be associated with each sample)
            self.patient_associations = self.patient_associations.loc[self.restricted_sample_ids, self.restricted_sample_ids]

            logging.info("Subsetted dataset from {} sample IDs to {} with at least {} hours of data\n\trestricted length of stay: {},\n\trestricted targets: {}\n\trestricted associations: {}".format(
                len(self.all_hourly_lengths), len(self.restricted_sample_ids), required_length, self.length_of_stay.shape, self.target.shape, self.patient_associations))
        else:
            self.restricted_sample_ids = self.sample_ids  # as iterable, __len__ and __getitem__ index into the restricted IDs 
            logging.info("Required length {} hours, did not subset the dataset by hourly length".format(required_length))

        # Recompute the class balances on the restricted sample IDs
        self.class_balance = self.precompute_class_balance()

    def precompute_class_balance(self, threshold=1.0):
        """ Precompute the classification balances for use in inverse-class-frequency cross-entropy loss function
            Note:  doesn't use self.target because those values are not filled until a full run through the dataset
        
        Args:
            threshold:  (float) number of days to compare remaining LoS to
        """
        logging.info("Pre-computing binary class balances from LoS with threshold of {}".format(threshold))
        if self.params.predict_after_hours > 0:
            # Single prediction per sample
            mask = (self.length_of_stay >= self.params.predict_after_hours/24.)  # require they haven't been discharged before prediction time
            remaining_los = (self.length_of_stay - self.params.predict_after_hours/24.)
            remaining_los = remaining_los.flatten()[mask.flatten()]
            class_targets = (remaining_los <= threshold)
        else:
            # Full time-series
            mask = (self.length_of_stay >= np.arange(self.params.pad_length)/24.)    # require they haven't been discharged before prediction time
            remaining_los = (self.length_of_stay - np.arange(self.params.pad_length)/24.)
            remaining_los = remaining_los.flatten()[mask.flatten()]
            class_targets = (remaining_los <= threshold)

        logging.info("All classification targets (masked & flattened):  {}".format(class_targets.shape))

        # Display LoS
        logging.info("Mean Length of Stay (shape {}):  {:.2f}+/-{:.2f}, min {:.2f}, max {:.2f}".format(
            self.length_of_stay.shape,
            np.mean(self.length_of_stay), np.std(self.length_of_stay),
            np.min(self.length_of_stay), np.max(self.length_of_stay)))
        # Display LoS
        logging.info("Mean Remaining Length of Stay (shape {}):  {:.2f}+/-{:.2f}, min {:.2f}, max {:.2f}".format(
            remaining_los.shape,
            np.mean(remaining_los), np.std(remaining_los),
            np.min(remaining_los), np.max(remaining_los)))

        # Compute binary class balances (sum to 1)
        class_balance = np.array([np.mean(class_targets), 1.-np.mean(class_targets)])
        logging.info("Class balances:  {}".format(class_balance))
        return class_balance


    def __len__(self):
        # logging.info("all samples, type {}:\n{}".format(type(self.restricted_sample_ids), self.restricted_sample_ids[:5]))
        return len(self.restricted_sample_ids)
    
    def __getitem__(self, index):
        """ Retrieve a single sample and bundle the data for subsequent batching (performed by DataLoader)

        Args:
            index: (int)  random index into len(self) generated by PyTorch/Skorch batcher
        """
        # Get encounter ID (CSN)
        sample_id = self.restricted_sample_ids[index]
        logging.debug("Retrieving sample:  {}".format(sample_id))
        
        # Get all associated patients
        # Defined as anyone in a unit with a batch patient simultaneously, or within params.association_history hours prior
        associated_patients, associated_patients_shared_time_index, associated_patient_time_scatter_mask, associated_patient_location_gather_index, associated_hourly_occupancies = self.get_associated_patients_sample(sample_id, self.patient_associations, self.hourly_patient_locations, self.all_locations)
        sample_associated_patient_count = np.array(len(associated_patients), dtype=np.int32) # convert for batching
        logging.debug("Found {} associated patients".format(len(associated_patients)))
        
        
        # Create mask/index for positions of batch patient(s) within batch-associated patients
        # No longer necessary if duplicating batch-associated data?
        sample_index = 0
        
        # Get all static data (cohort-level, e.g. non-hourly)
        sample_static_features = self.cohort_df.loc[associated_patients, :].values.astype(np.float32)
        logging.debug("Associated patients' static features: {}".format(sample_static_features.shape))
        
        # Pad the static features if less than maximum associated patients
        if len(sample_static_features) < self.params.max_associated_patients:
            pad_array = np.full(shape=(self.params.max_associated_patients - len(sample_static_features),
                                       sample_static_features.shape[1]),
                                fill_value=self.params.pad_value,
                                dtype=np.float32)
            # Pad static features to fixed shape per-sample
            logging.debug("Concatenating static features {} with pad array {}".format(sample_static_features.shape, pad_array.shape))
            sample_static_features = np.concatenate((sample_static_features,
                                                     pad_array),
                                                    axis=0)
            
        # Pad the associated patient time mask to a fixed number of associated patients
        # Reshape to associated patients x time steps
        # Convert hourly time mask to numpy / uint8 for batching
        # Boolean would be ideal if ever implemented for PyTorch types
        sample_hourly_timemask = np.zeros(shape=(self.params.max_associated_patients, self.params.associated_time_max),
                                          dtype=np.uint8)
        sample_hourly_timemask[:int(sample_associated_patient_count), :] = associated_patient_time_scatter_mask.values.astype(np.uint8).T
        
        # Reshape the hourly occupancies and convert to numpy for batching
        sample_hourly_occupancies = associated_hourly_occupancies.values.astype(np.int32).T
        logging.debug("Hourly Occupancies over associated patients:  {}\n{}".format(sample_hourly_occupancies.shape, sample_hourly_occupancies))
        
        # Bundle all hourly data for associated patients
        # Initialize sample arrays
        logging.debug("Reformatting hourly features...")
        sample_hourly_features = np.full(fill_value=self.params.pad_value,
                                          shape=(self.params.max_associated_patients, self.params.pad_length, len(self.params.hourly_features)),
                                          dtype=np.float32)
        sample_hourly_lengths = np.zeros(self.params.max_associated_patients, dtype=np.int32)
        
        # Fill sample arrays:  Pad and mask time dimension to a uniform shape
        for asi, associated_sample_id in enumerate(associated_patients):
            associated_hourly_df = self.hourly_dataframes[associated_sample_id].loc[:, self.hourly_features]
            valid_shared_index = associated_hourly_df.index.intersection(associated_patients_shared_time_index)
            logging.debug("Associated sample {} hourly DF:  {}".format(associated_sample_id, associated_hourly_df.shape))
            masked_hourly_df = associated_hourly_df.loc[valid_shared_index, :]
            logging.debug("Portion of associated hourly DF in the shared time index:  {}".format(masked_hourly_df.shape))
            sample_hourly_lengths[asi] = min(self.params.pad_length, len(masked_hourly_df))
            if len(masked_hourly_df) > self.params.pad_length:
                logging.debug("Pad length {} insufficient to hold {} hourly data for associated patient {} in shared time index {}".format(
                    self.params.pad_length, masked_hourly_df.shape, associated_sample_id, associated_patients_shared_time_index.shape))
            logging.debug("Sample pad length: {}".format(sample_hourly_lengths[asi]))
            clipped_df = masked_hourly_df.iloc[:sample_hourly_lengths[asi], :]
            logging.debug("Hourly data clipped to pad length:  {}\n{}".format(clipped_df.shape, clipped_df[:3]))
            sample_hourly_features[asi, :sample_hourly_lengths[asi], :] = clipped_df
            logging.debug("All data clipped and padded to pad length:  {}\n{}".format(sample_hourly_features.shape, sample_hourly_features[:3]))
        logging.debug("Associated patients' hourly features: {}\n{}".format(sample_hourly_features.shape, sample_hourly_features[:3]))
                
        # Create gather index to re-gather the per-sample hourly data from where it was scattered in shared time index
        sample_hourly_time_gather_index = np.zeros(shape=(self.params.pad_length), dtype=np.int64)
        patient_start_in_shared_index = self.params.associated_time_max = self.params.pad_length
        patient_end_in_shared_index = min(patient_start_in_shared_index + sample_hourly_lengths[0], self.params.associated_time_max)
        patient_len_in_shared_index = patient_end_in_shared_index - patient_start_in_shared_index  # how much of the sample_id hourly data falls within the shared time index
        sample_hourly_time_gather_index[:patient_len_in_shared_index] = np.arange(patient_start_in_shared_index, patient_end_in_shared_index)
        
        # Create mask for associated patient/unit location matching
        # Allows scattering the short-term hourly data on a per-sample timescale to the global time indices
        # Drop head and tail of all NaN (leftover global time indices for non-associated patients)
        sample_associated_patient_location_gather_index = associated_patient_location_gather_index
        
        # Create index into all_locations at each timestep for the sample_id only
        sample_hourly_locations = pd.Series(0, index=associated_patients_shared_time_index)
        valid_shared_index = associated_patients_shared_time_index.intersection(self.hourly_patient_locations)
        sample_hourly_locations.loc[valid_shared_index] = self.hourly_patient_locations.loc[valid_shared_index, sample_id]
        logging.debug("Sample {} associated patient hourly locations:  {}\n{}".format(sample_id, sample_hourly_locations.shape, sample_hourly_locations[:5]))
        sample_hourly_locations = sample_hourly_locations.replace(self.all_locations_index_dict)  # default location to 0 to prevent errors on gather, should be masked later
        sample_hourly_locations = sample_hourly_locations.values.astype(np.int64)
        
        # Create index into all_locations at each timestep for the associated patients
#         sample_hourly_locations = pd.DataFrame(0, index=associated_patients_shared_time_index, columns=associated_patients)
#         valid_shared_index = associated_patients_shared_time_index.intersection(self.hourly_patient_locations)
#         sample_hourly_locations.loc[valid_shared_index, associated_patients] = self.hourly_patient_locations.loc[valid_shared_index, associated_patients]
#         logging.debug("Sample {} associated patient hourly locations:  {}\n{}".format(sample_id, sample_hourly_locations.shape, sample_hourly_locations[:5]))
#         sample_hourly_locations = sample_hourly_locations.replace(self.all_locations_index_dict)  # default location to 0 to prevent errors on gather, should be masked later
#         sample_hourly_locations = sample_hourly_locations.values.astype(np.int64).T
#         if len(associated_patients) < self.params.max_associated_patients:
#             sample_hourly_locations = np.concatenate((sample_hourly_locations,
#                                                       np.zeros(shape=((self.params.max_associated_patients - len(associated_patients)),
#                                                                  sample_hourly_locations.shape[1]),
#                                                           dtype=np.int64)),
#                                                 axis=0)
        logging.debug("Hourly location indices:  {}\n{}".format(sample_hourly_locations.shape, sample_hourly_locations[:5]))    
    
    
        logging.debug("Associated Patients:  {},\tStatic Features:  {},\tHourly Locations:  {},\tHourly Time-Gather Index:  {},\tHourly Features:  {},\tHourly Lengths:  {},\tHourly Masks:  {},\tHourly Occupancies:  {}".format(
            sample_associated_patient_count,
            sample_static_features.shape,
            sample_hourly_locations.shape,
            sample_hourly_time_gather_index.shape,
            sample_hourly_features.shape,
            sample_hourly_lengths.shape,
            sample_hourly_timemask.shape,
            sample_hourly_occupancies.shape
        ))
        
        # Bundle all arrays/values for batching, and return
        sample = (sample_associated_patient_count, sample_static_features, sample_hourly_locations, sample_hourly_time_gather_index, associated_patient_location_gather_index, sample_hourly_features, sample_hourly_lengths, sample_hourly_timemask, sample_hourly_occupancies)
        
        # Create the label
        if self.params.predict_remaining_LoS:
            if self.params.predict_after_hours == 0:
                yi = np.zeros(shape=(self.params.pad_length), dtype=np.float32)
                yi[:sample_hourly_lengths[sample_index]] = self.length_of_stay[index] - np.arange(sample_hourly_lengths[sample_index])/24.
            else:
                assert self.params.predict_after_hours <= self.params.pad_length, "Must have more that {} hour pad length to predict after {} hours".format(self.params.pad_length, self.params.predict_after_hours)
                yi = self.length_of_stay[index] - self.params.predict_after_hours/24. # predict (remaining) LoS in days at each timestep
                # self.target[index] = yi  # save for easy lookup later
            # yi[:self.params.pad_length] = self.Y[index] - np.arange(self.params.pad_length)/24.
        else:
            if self.params.predict_after_hours == 0:
                yi = np.zeros(shape=(self.params.pad_length), dtype=np.float32)
                yi[:sample_hourly_lengths[sample_index]] = self.length_of_stay[index] # single-value prediction in days
            else:
                yi = self.length_of_stay[index]

        # Choose model output type
        if self.params.output_type == "regression":
            # Use the defined yi as targets (remaining LoS or LoS)
            pass
        elif self.params.output_type == "classification":
            # Classification label is 1 if discharge within 1 day, or 0 if discharged after
            # Compare total or remaining LoS to 1day=24hr, but keep datatype/size for targets
            N_sample_output_dims = len(yi.shape)
            yi = (yi <= 1.) 

            # Convert to one-hot output
            yi = np.expand_dims(yi, N_sample_output_dims) # add a new dimension at end
            yi = np.concatenate((yi, ~yi), axis=(N_sample_output_dims)).astype(np.float32)
        else:
            logging.warning("Unknown output_type:  {}".format(self.params.output_type))
            raise NotImplementedError
        
        # save yi targets for easy lookup later
        if self.params.predict_after_hours >= 0:
            self.target[index] = yi  
        else:
            self.target[index, :] = yi
        return sample, yi


if __name__ == '__main__':
    """ Run a unit test to validate the dataloader
    """

    utils.set_logger("./log/dataloader.log", logging.INFO)

    # Create test parameters
    params = utils.Params(
        {
            "output_type": "classification",  # Classification vs Regression
            "project_dir": "..",  # Top-level path for this project
            "data_dir": "Data",  # Relative data path
            "cohort_file": "itan_cohort_v.h5",  # File path to the cohort dataset (hdf5)
            "sample_dir": "Data/itan_hourly_encounter_splits",  # Directory to the split patient hourly data
            "label": "LOS",  # Column name of target/label in cohort dataframe
            "cohort_features": ["ADM_LAPS2", "ADM_COPS2", "SEX", "AGE"],  # Which cohort-level features to use
            "hourly_features": ["LAPS2"],#, "IMAR_IM_GROUP"],  # Which patient hourly features to use
            "all_locations": ['OR', 'WARD', 'ICU', 'TCU', 'PAR'], # ICU, WARD, OR / 
            "pad_length": 96,  # maximum number of hours of each time-series to include
            "association_history": 0,  # How many hours to look back in unit occupancy to define patient associations
            "associated_time_max": 24,  # maximum number of hours to consider over all all associate patients
            "max_associated_patients": 100,  # maximum number of associated patients to consider over all time
            "max_occupancy": 100,  # maximum number of associated patients to consider in a unit at a single time
            "pad_value": 0,  # Value to pad variable-length input sequences with (unused), and to fill NaNs
            "hourly_embedding_fill_value": 0,  # Value to pad variable-length input sequences with (unused), and to fill NaNs
            "patient_hourly_hidden_size": 32,  # hidden size of LSTM output
            "unit_hourly_hidden_size1": 64,  # hidden size of LSTM output when merging patient hourly data over unit occupancy
            "unit_hourly_hidden_size2": 128,  # hidden size of LSTM output when accumulating unit embeddings over time
            "num_hourly_output_features": 32,  # hidden size of LSTM output
            "num_layers": 5,  # number of stacked LSTM layers (each with same hidden_size)
            "model_name": "dataloader_debug",  # which model to train ('linear', 'lstm')
            "included_features": ["static", "hourly", "strain"],  # Whether to feed static, hourly, and/or strain features directly into the output layer
            "predict_remaining_LoS": True,  # Whether to predict remaining LoS instead of total LoS
            "predict_after_hours": 0,  # Number of hours to index the time series target and input.  0 to predict at every point.  use included_features parameter to restrict to static data
            "num_epochs": 5,  # number of training epochs.  Note:  can rerun the training cell if warm_start=True
            "batch_size": 5,  # number of samples per batch.  Usually limited by GPU memory.
            "learning_rate": 0.01,  # optimizer learning rate
            "dropout_prob": 0.0,  # dropout regularization probability (0 for not implemented)
            "cross_validation_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},  # Fraction of data in each CV split
            "split_file": "set_splits.json",  # Where to save the CV splits
            "subset": 0,  # train/evaluate on a smaller subset of the data for testing purposes, reduced runtime
            "torch_random_seed": 7532941,  # RNG seed for PyTorch (weight initialization and dataset shuffling per-epoch)
            "numpy_random_seed": 7532941,  # RNG seed for NumPy (randomly selecting among associated occupants at each hour)
            "normalize_cohort": True,  # Whether to normalize the cohort features (mean 0, std 1)
            "normalize_hourly": True,  # Whether to normalize the hourly features (mean 0, std 1), rolling computation over all sample IDs
        }
    )

    model_dir = os.path.join(params.project_dir, 'Code/experiments/', params.model_name)

    split_filepath = os.path.join(params.project_dir, "Code/", "experiments/", params.model_name, params.split_file)
    cv_split_samples = utils.load_dict_from_json(split_filepath)
    logging.info("Loaded CV splits from {}".format(split_filepath))
    for split in cv_split_samples:
        logging.info("{} split:  {} samples".format(split, len(cv_split_samples[split])))

    # Set the PyTorch seed for shuffline and weight initialization reproducibility
    torch.manual_seed(params.torch_random_seed)
    np.random.seed(params.numpy_random_seed)

    # Construct Skorch datasets for training
    logging.info("Constructing Strain datasets for train cross-validation split.")
    datasets = {}
    split = "train"
    logging.info("\nBuilding dataset for {} split".format(split))
    datasets[split] = ITANStrainDataset(cv_split_samples[split], params, mode=split, show_progress_bar=False)
    logging.info("{} split:  {} samples:\n{}\nX: {}\nY: {}".format(split, len(cv_split_samples[split]), cv_split_samples[split][:3], None, datasets[split].length_of_stay.shape))
    
    logging.info("LoS values:  {}:  mean {:.2f} +/- {:.2f}\n{}".format(
        datasets[split].length_of_stay.shape,
        np.mean(datasets[split].length_of_stay),
        np.std(datasets[split].length_of_stay),
        datasets[split].length_of_stay))

    logging.debug("Initialized target values:  {}:  mean {:.2f} +/- {:.2f}\n{}".format(
        datasets[split].target.shape,
        np.mean(datasets[split].target),
        np.std(datasets[split].target),
        datasets[split].target))
    
    # Test iteration through dataset
    with tqdm(total=len(datasets[split]), desc="Testing dataset iterator") as st:
        for sample in datasets[split]:
            X, y = sample
            # logging.info("Sample:\n{}".format(sample))
            logging.info("\nSingle sample target (shape {}):\n{}".format(y.shape, y))
            st.update()
            break

    split="val"
    logging.info("\nBuilding dataset for {} split".format(split))
    # Note use the normalization coefficients from the training split
    datasets[split] = ITANStrainDataset(cv_split_samples[split], params, mode=split, trained_params=datasets["train"].trained_params)
                
    # Update list of cohort features with categorical expansion
    params.expanded_cohort_features = datasets[split].cohort_features
