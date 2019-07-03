# Basic utilities
import datetime
import json
import pickle
import math
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import logging
import argparse
import os
from pprint import pprint, pformat

# Deep Learning
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# Local files
import utils 


# Neural Network definition
class LinearRegressorModule(nn.Module):
    def __init__(self, params, num_units=10, nonlin=F.relu, ):
        super().__init__()

        # Define network layers
        self.denseX = nn.Linear(len(params.hourly_features)+len(params.cohort_features)+1, 1) #*params.pad_length
        
        # Manual initialization
        self.denseX.weight.data.fill_(0.0)
        self.denseX.bias.data.fill_(1.0)

    def forward(self, X, **kwargs):
        shape_debug = False
        if shape_debug: logging.info("input: {}\n{}".format(X.size(), X.dtype))
        X = torch.mean(X, dim=1, keepdim=False)
        if shape_debug: logging.info("reshape: {}".format(X.size()))
        X = self.denseX(X)
        if shape_debug: logging.info("output: {}".format(X.size()))
        return X
    

# Recurrent Neural Network definition
class LSTMRegressorModule(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_size = len(params.hourly_features)+len(params.cohort_features)+1
        hidden_size = params.hidden_size
        
        # Define network layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.output_regression = nn.Linear(hidden_size, 1)
        
#         self.bn1 = 

    def forward(self, inputs):
        shape_debug = False
        sequences, lengths = inputs
        if shape_debug: logging.info("input:  {}".format(sequences.size()))
        if shape_debug: logging.info("lengths:  {}".format(lengths.size()))
        
        output, (h_n, c_n) = self.lstm1(sequences)
        X = output
        if shape_debug: logging.info("LSTM 1: {}".format(X.size()))
            
        length_indices = lengths.view(-1, 1, 1).expand(X.size(0), 1, X.size(2)).long() - 1
        if shape_debug: logging.info("Slice Indices:  {}".format(length_indices.size()))
#         if shape_debug: logging.info("Slice Indices:  {}\n{}".format(length_indices.size(), length_indices))
        
#         X = torch.index_select(output, dim=1, index=lengths)
        X = X.gather(1, length_indices)
        if shape_debug: logging.info("Length-indexed outputs:  {}".format(X.size()))
            
        X = X.squeeze(1)
        if shape_debug: logging.info("Last Output in Sequence:  {}\n{}".format(X.size(), X.dtype))
        
        X = self.output_regression(X)
        if shape_debug: logging.info("Output: {}".format(X.size()))
        return X
    
class ITANStrainNetworkModule(nn.Module):
    def __init__(self, params):
        """ Load parameters and initialize NN model
        
        Params:
            output_type:  (string) target type, determines output layer
            include_strain_variables:  (bool) whether to use the unit and network-level strain connections
            hourly_features:  (list) list of hourly feature column names
            cohort_features:  (list) list of cohort-level (on admission) feature column names
            patient_hourly_hidden_size:  (int) hidden size of LSTM for all batch-associated patient hourly features
            unit_hourly_hidden_size:  (int) hidden size of set accumulator for embedding all patients simultaneously occupying a unit
            num_hourly_output_features:  (int)
        """
        super().__init__()
        # Compute and save parameters
        self.params = params
        self.input_size = len(self.params.hourly_features)

        # Compute the size of the output layer based on which features are utilized
        self.num_output_features = 0
        if "static" in self.params.included_features:
            self.num_output_features += len(self.params.expanded_cohort_features)
        if "hourly" in self.params.included_features:
            # self.num_output_features += len(self.params.hourly_features)
            self.num_output_features += self.params.patient_hourly_hidden_size
        if "strain" in self.params.included_features:
            self.num_output_features += self.params.num_hourly_output_features
        if "elapsed" in self.params.included_features:
            self.num_output_features += 1
        
        # Define internal network layers and RNN cells
        self.lstm_associated_patient = nn.LSTM(input_size=self.input_size,
                                               hidden_size=self.params.patient_hourly_hidden_size,
                                               num_layers=1,
                                               batch_first=True,
                                               bidirectional=False)
        self.associated_patient_activation = torch.nn.LeakyReLU()
        
        self.patient_set_accumulate = nn.LSTM(input_size=self.params.patient_hourly_hidden_size,
                                              hidden_size=self.params.unit_hourly_hidden_size1,
                                              num_layers=1,
                                              batch_first=True,
                                              dropout=params.dropout_prob,
                                              bidirectional=False)#True)
        self.set_accumulate_activation = torch.nn.LeakyReLU()

        self.lstm_unit = nn.LSTM(input_size=self.params.unit_hourly_hidden_size1,
                                 hidden_size=self.params.unit_hourly_hidden_size2,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False)
        self.unit_activation = torch.nn.LeakyReLU()
        
        self.lstm_sample_patient = nn.LSTM(input_size=self.params.unit_hourly_hidden_size2+self.input_size,
                                           hidden_size=self.params.num_hourly_output_features,
                                           num_layers=1,
                                           batch_first=True,
                                           bidirectional=False)
        self.sample_patient_activation = torch.nn.LeakyReLU()
        
        # Define output layer
        if self.params.output_type == "classification":
            self.output_layer = nn.Linear(self.num_output_features, 2)
            self.classifier = nn.LogSoftmax(dim=2)
        elif self.params.output_type == "regression":
            self.output_layer = nn.Linear(self.num_output_features, 1)
        else:
            logging.warning("Unknown output type:  {}".format(self.output_layer))
            
        
    def forward(self, inputs):
        """ Inputs:  tuple built by data_loader
        
        Def:  Batch patients/samples:  Exactly batch_size randomly selected set of patients to make an output prediction for the current batch.  
        Def:  Patients associated with batch:  Unknown number of patients who were in any unit that one of the batch patients eventually occupied
              This allows for building a unit history prior to the batch patients' occupancy
              Always at least batch_size, additional patients may be clipped to a randomly selected subset for size restrictions and regularization
        Def:  pad_length:  maximum number of hours of data for any patient in the batch
        Def:  associated_time_max:  Total number of hours from first sample of any batch-associated patient to last sample of any
        
        Args:
            inputs:  (tuple)  all batch data generated per-sample by data loader and batched by Skorch
            batch_samples:  (batch_patients)  boolean mask for which patients are being predicted out of all patients associated with the batch
                            sums to batch_size
            patient_inputs:  (tuple)  all data for patients associated with the batch
            
            patient_static_features:  (batch_patients x num_patient_static_features)  static features for all patients associated with the batch
            Xpatient_locations:  (batch_patients x time_max x num_units)  location of each patient associated with the batch at each time (global time)
            unit_hourly_occupants:  (batch_units x time_max x max_occupancy)  indices into batch patients for all occupants of each unit at each global time
            patient_hourly_features:  (batch_patients x pad_length x num_patient_hourly_features)  full history of patient hourly features for each patient associated with batch
            patient_hourly_lengths:  (batch_patients)  number of valid samples in hourly sequences (clipped to pad_length)
            patient_hourly_timemask:  (batch_patients x time_max)  boolean mask to map personal time indices to global time index
        """
        shape_debug = False
        # Extract bundled batch data
        patient_association_count, patient_static_features, patient_hourly_locations, patient_hourly_time_gather_index, associated_patient_location_gather_index, patient_hourly_features, patient_hourly_lengths, patient_hourly_timemask, patient_hourly_occupancies = inputs
        sample_batch_size = patient_static_features.size()[0]
        if shape_debug: logging.info("Batch Static Features: {}, type {}".format(patient_static_features.size(), patient_static_features.dtype))
        if shape_debug: logging.info("Batch Hourly Locations (gather index): {}, type {}".format(patient_hourly_locations.size(), patient_hourly_locations.dtype))
        if shape_debug: logging.info("Batch Hourly Time-Gather Index (gather index): {}, type {}".format(patient_hourly_time_gather_index.size(), patient_hourly_time_gather_index.dtype))
        if shape_debug: logging.info("Batch Associated Patient Locations Gather Index:  {}".format(associated_patient_location_gather_index.shape))
        if shape_debug: logging.info("Batch Hourly Features: {}".format(patient_hourly_features.size()))
        if shape_debug: logging.info("Batch Hourly Lengths: {}".format(patient_hourly_lengths.size()))
        if shape_debug: logging.info("Batch Hourly Shared-Time Masks (scatter mask): {}".format(patient_hourly_timemask.size()))
        if shape_debug: logging.info("Batch Hourly Occupancies: {}".format(patient_hourly_occupancies.size()))

        # Build list of features based on desired model complexity
        patient_all_features = []
        if "static" in self.params.included_features:
            # Tile and append patient static features
            patient_static_features = patient_static_features[:,0,:]  # select out the sample patients from associated for each in batch
            patient_static_features = patient_static_features.unsqueeze(1).repeat(1, self.params.pad_length, 1)
            if shape_debug: logging.info("Batch Static Features tiled for concat to final features: {}, type {}".format(patient_static_features.size(), patient_static_features.dtype))
            patient_all_features += [patient_static_features]
        # if "hourly" in self.params.included_features:
        #     patient_all_features += [patient_hourly_features[:,0,:,:]]
        if "elapsed" in self.params.included_features:
            patient_elapsed_LoS = torch.arange(self.params.pad_length).float().repeat((patient_static_features.size()[0], 1)).unsqueeze(2)
            if shape_debug: logging.info("Batch Elapsed LoS out to pad length: {}".format(patient_elapsed_LoS.size()))
            patient_all_features += [patient_elapsed_LoS]
        if ("hourly" in self.params.included_features) or ("strain" in self.params.included_features):
            # LSTM Expects (batch, seq, feature) with batch_first=True
            # Need to combine the batch and associated patient dimensions then reconstruct after
            patient_hourly_features = patient_hourly_features.view(-1,
                                                                   self.params.pad_length,
                                                                   self.input_size)
            if shape_debug: logging.info("Batch Hourly Features reshaped for LSTM: {}".format(patient_hourly_features.size()))

            # Wrap up time-history for each patient associated with batch
            # Keep full RNN history (1-directional)
            patient_personal_hourly_embeddings, _ = self.lstm_associated_patient(patient_hourly_features)  # (batch_patients x pad_length x patient_hourly_hidden_size)
            if shape_debug: logging.info("Patient Hourly Embeddings personal-time: {}".format(patient_personal_hourly_embeddings.size()))
            patient_personal_hourly_embeddings = patient_personal_hourly_embeddings.view(sample_batch_size,
                                                                                         self.params.max_associated_patients,
                                                                                         self.params.pad_length,
                                                                                         self.params.patient_hourly_hidden_size)
            if shape_debug: logging.info("Patient Hourly Embeddings personal-time reshaped to batches: {}".format(patient_personal_hourly_embeddings.size()))
            patient_personal_hourly_embeddings = self.associated_patient_activation(patient_personal_hourly_embeddings)

            patient_all_features += [patient_personal_hourly_embeddings[:,0,:,:]]

        if "strain" in self.params.included_features:
            # Get batch-associated patient hourly embeddings on global time index, index out patients for each unit at each time
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            patient_global_hourly_embeddings = torch.full(size=(sample_batch_size,
                                                                self.params.max_associated_patients,
                                                                self.params.associated_time_max,
                                                                self.params.patient_hourly_hidden_size),
                                                          fill_value=self.params.hourly_embedding_fill_value,
                                                          dtype=torch.float32)
            patient_global_hourly_embeddings.masked_scatter_(patient_hourly_timemask.unsqueeze(3), patient_personal_hourly_embeddings)  # (batch_patients x time_max x patient_hourly_hidden_size)
            if shape_debug: logging.info("Patient Hourly Embeddings scattered to global-time: {}".format(patient_global_hourly_embeddings.size()))
                
            # Group batch-associated patient hourly embeddings by simultaneous unit occupancy and expand dimension appropriately
            # gather:  index shape is same as output shape (gather does not broadcast)
            # index provided by data loader
            patient_global_hourly_embeddings = patient_global_hourly_embeddings.unsqueeze(1).repeat((1, associated_patient_location_gather_index.size()[1], 1, 1, 1))
            if shape_debug: logging.info("Patient Hourly Embeddings tiled for unit-gather: {}".format(patient_global_hourly_embeddings.size()))
            associated_patient_location_gather_index = associated_patient_location_gather_index.unsqueeze(4).repeat((1, 1, 1, 1, patient_global_hourly_embeddings.size()[4]))
            if shape_debug: logging.info("Batch A-P Hourly Locations Gather Index tiled for unit-gather: {}".format(associated_patient_location_gather_index.size()))
            unit_patient_hourly_embeddings = patient_global_hourly_embeddings.gather(dim=2, index=associated_patient_location_gather_index)
            if shape_debug: logging.info("Patient Hourly Embeddings gathered by unit occupancy: {}".format(unit_patient_hourly_embeddings.size()))
        
            # Merge hourly embeddings over all batch-associated patients in a unit simultaneously
            # Can be done with reduce-max, RNN, shuffle/RNN, or search literature for unordered set combination methods
            unit_patient_hourly_embeddings = unit_patient_hourly_embeddings.permute(0, 1, 3, 2, 4)
            if shape_debug: logging.info("Unit-Patient Hourly Embeddings permuted for reshaping: {}".format(unit_patient_hourly_embeddings.size()))
            unit_patient_hourly_embeddings = unit_patient_hourly_embeddings.contiguous().view(-1, self.params.max_occupancy, self.params.patient_hourly_hidden_size)
            if shape_debug: logging.info("Unit-Patient Hourly Embeddings reshaped for merging over occupants (LSTM): {}".format(unit_patient_hourly_embeddings.size()))
            output, (h_n, c_n) = self.patient_set_accumulate(unit_patient_hourly_embeddings) 
            if shape_debug: logging.info("Raw set-accumulation output: {} full, {} final".format(output.size(), c_n.size()))
            patient_hourly_occupancies = patient_hourly_occupancies.view(-1)
            if shape_debug: logging.info("Patient Hourly Occupancies reshaped for for indexing LSTM output: {}".format(patient_hourly_occupancies.size()))
            if shape_debug: logging.info("Occupancies from {} to {}".format(patient_hourly_occupancies.min(), patient_hourly_occupancies.max()))
            occupancy_indices = patient_hourly_occupancies.view(-1, 1, 1).expand(output.size(0), 1, output.size(2)).long() - 1
            if shape_debug: logging.info("Occupancy indices: {}".format(occupancy_indices.size()))
            if shape_debug: logging.info("Occupancy indices from {} to {}".format(occupancy_indices.min(), occupancy_indices.max()))
            unit_hourly_embeddings = output.gather(1, occupancy_indices)
            if shape_debug: logging.info("Unit Hourly Embeddings merged over occupants: {}".format(unit_hourly_embeddings.size()))
            unit_hourly_embeddings = unit_hourly_embeddings.view(-1, len(self.params.all_locations), self.params.associated_time_max, self.params.unit_hourly_hidden_size1)
            unit_hourly_embeddings = self.set_accumulate_activation(unit_hourly_embeddings)
            if shape_debug: logging.info("Unit Hourly Embeddings reshaped to match input: {}".format(unit_hourly_embeddings.size()))
            
            # Wrap up time-history for each unit associated with batch
            # Unit summaries change as patients move in and out
            # Keep full history
            unit_hourly_embeddings = unit_hourly_embeddings.view(-1, self.params.associated_time_max, self.params.unit_hourly_hidden_size1)
            if shape_debug: logging.info("Unit Hourly Embeddings reshaped for accumulating over time (LSTM): {}".format(unit_hourly_embeddings.size()))
            unit_accumulated_embeddings, _ = self.lstm_unit(unit_hourly_embeddings)
            if shape_debug: logging.info("Unit Accumulated Embeddings from LSTM: {}".format(unit_accumulated_embeddings.size()))
            unit_accumulated_embeddings = unit_accumulated_embeddings.view(-1, len(self.params.all_locations), self.params.associated_time_max, self.params.unit_hourly_hidden_size2)
            if shape_debug: logging.info("Unit Accumulated Embeddings reshaped for gathering: {}".format(unit_accumulated_embeddings.size()))
            unit_accumulated_embeddings = self.unit_activation(unit_accumulated_embeddings)

            # Select out the unit-level data for each batch patient at each global time
            # Create gather index from patient locations
            patient_hourly_locations = patient_hourly_locations.unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, unit_accumulated_embeddings.size()[3])
            if shape_debug: logging.info("Patient Hourly Locations tiled and reshaped as a gather index: {}".format(patient_hourly_locations.size()))
            patient_global_hourly_unit_features = unit_accumulated_embeddings.gather(dim=1, index=patient_hourly_locations)
            if shape_debug: logging.info("Patient Unit Hourly Embeddings gathered from hourly locations in global-time: {}".format(patient_global_hourly_unit_features.size()))
            patient_global_hourly_unit_features = patient_global_hourly_unit_features.squeeze(1)
            if shape_debug: logging.info("Patient Unit Hourly Embeddings squeezed to drop location dimension: {}".format(patient_global_hourly_unit_features.size()))
                
            # Re-index to personal (sample) time for each batch patient
            patient_hourly_time_gather_index = patient_hourly_time_gather_index.unsqueeze(2).repeat(1, 1, self.params.unit_hourly_hidden_size2)
            if shape_debug: logging.info("Hourly Time-Gather Index tiled for personal time gather: {}, type {}".format(patient_hourly_time_gather_index.size(), patient_hourly_time_gather_index.dtype))
            patient_personal_hourly_unit_features = patient_global_hourly_unit_features.gather(dim=2, index=patient_hourly_time_gather_index)
            if shape_debug: logging.info("Patient Unit Hourly Embeddings in personal-time: {}".format(patient_personal_hourly_unit_features.size()))
                
            # Select out the hourly features for the sample patients out of all associated patients for each sample in batch
            patient_hourly_features = patient_hourly_features.view(-1, self.params.max_associated_patients, self.params.pad_length, self.input_size) # fix to original shape
            patient_hourly_features = patient_hourly_features[:,0,:,:]#.squeeze(1)
            if shape_debug: logging.info("Patient Hourly Features (for sample_id only): {}".format(patient_hourly_features.size()))
                
            # Merge patient hourly and the unit data for their current location
            patient_hourly_features = torch.cat((patient_hourly_features,
                                                 patient_personal_hourly_unit_features),
                                                dim=2)
            if shape_debug: logging.info("Patient Hourly Features (unit/strain & personal merged): {}".format(patient_hourly_features.size()))
            
            # Wrap up time-history for each patient as they move through units
            output, (h_n, c_n) = self.lstm_sample_patient(patient_hourly_features)
    #         length_indices = patient_hourly_lengths.view(-1, 1, 1).expand(output.size(0), 1, output.size(2)).long() - 1
    #         patient_accumulated_features = output.gather(1, length_indices)  # to select only the last valid timestep
            patient_accumulated_features = output
            if shape_debug: logging.info("Patient Hourly Embeddings (wrap all features over personal time): {}, type {}".format(patient_accumulated_features.size(), patient_accumulated_features.dtype))
            patient_accumulated_features = self.sample_patient_activation(patient_accumulated_features)

            patient_all_features += [patient_accumulated_features]

        # Concatenate the static, hourly, and/or strain features into a single tensor to be fed to the output layer
        patient_all_features = torch.cat(patient_all_features, dim=2)
        if shape_debug: logging.info("All Patient Features (hourly embeddings with strain embeddings and static features): {}".format(patient_all_features.size()))
            
        # Make a prediction (regression/classification) for each patient in batch (at each time-elapsedLoS?)
        patient_hourly_pred = self.output_layer(patient_all_features)
        if self.params.output_type == "regression":
            patient_hourly_pred = patient_hourly_pred.squeeze(2)
        if shape_debug: logging.info("Patient Predictions at each personal-timestep: {}".format(patient_hourly_pred.size()))

        # Restrict to predicting at a single timestep
        if self.params.predict_after_hours > 0:
            pred_length = torch.full(size=patient_hourly_lengths[:, 0].size(),
                                     fill_value=1,
                                     dtype=torch.int64)

            # # Select the single prediction
            # if self.params.included_features == ["static"]:
            #     # Static data starts at index 0 before any hourly data exists
            #     pred = patient_hourly_pred[0]
            # else:
            #     # Hourly data starts at index 0 for after 1 hour
            #     # predict_after_hours starts at 1 if restricting to single-timestep output
            #     if self.params.predict_after_hours <= patient_hourly_lengths[:, 0]:
            #         # If have sufficient input data to predict at desired timestep
            #         pred = patient_hourly_pred[self.params.predict_after_hours - 1]
            #     else:
            #         # If not sufficient input data, extrapolate past the last valid prediction
            #         extend_prediction = self.params.predict_after_hours - patient_hourly_lengths[:, 0]
            #         pred = patient_hourly_pred[patient_hourly_lengths[:, 0]] - extend_prediction
            pred = patient_hourly_pred[:, (self.params.predict_after_hours - 1)].unsqueeze(1)
            # for now, assume pred_after_hours is less than the valid length.
            # otherwise need some fancy min/thresholding
        else:
            pred = patient_hourly_pred
            pred_length = patient_hourly_lengths[:, 0]

        if self.params.output_type == "classification":
            pred = self.classifier(pred)
            if shape_debug: logging.info("Classification predictions: {}".format(pred.size()))
        return pred, pred_length

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.

    Does not work with batched lengths
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class MSELossMasked(torch.nn.Module):
    """ Similar to PyTorch's MSELoss, but masked by the lengths of each sample in the batch
    """
    def __init__(self):
        super(MSELossMasked, self).__init__()

    def forward(self, input, target):
        """
        Params:
            input = (prediction, lengths)
            prediction:  (batch_size, pad_length) Tensor of predicted values for the batch
            target:  (batch_size, pad_length) Tensor of target values for the batch
            lengths: (batch_size)  Tensor of valid lengths for each sequence in the batch

        Returns:
            loss:  (Float) loss over the current batch
        """
        shape_debug = False
        prediction, lengths = input
        pad_length = prediction.size()[1]
        if shape_debug: logging.info("Loss Predictions: {}".format(prediction.size()))
        if shape_debug: logging.info("Loss Targets: {}".format(target.size()))
        if shape_debug: logging.info("Loss Lengths: {}".format(lengths.size()))
        if shape_debug: logging.info("Loss Pad Length: {}".format(pad_length))
        diffs = prediction - target

        # batch_ranges = torch.arange(pad_length, device=lengths.device,
        #                             dtype=lengths.dtype).expand(len(length), max_len) < length.unsqueeze(1)

        # Mask and compute SE over valid subsequences for all sequences in the batch
        # Sets squared-error to zero past the valid lengths
        length_mask = length_to_mask(lengths.view(-1), pad_length, dtype=torch.float32).view(-1, pad_length)
        squared_errors = torch.pow(torch.mul(diffs, length_mask), 2)
        
        # Average the errors over sequence length independently for each sample in the batch
        mean_errors = torch.div(torch.sum(squared_errors, dim=1, keepdim=False),
                                lengths.float())
        
        # Averaged over all samples in the batch so loss is invariant of batch size
        loss = torch.mean(mean_errors)

        return loss



class CrossEntropyLossMasked(torch.nn.Module):
    """ Similar to PyTorch's CrossEntropyLoss, but masked by the lengths of each sample in the batch
    """
    def __init__(self):
        super(CrossEntropyLossMasked, self).__init__()

    def forward(self, input, target):
        """
        Computes the cross-entropy loss function from the binary predictions and log-probabilities

        CE = −(ylog(p)+(1−y)log(1−p))
           = −ylog(p)      --for binary output, summed over samples and masked by valid lengths

        Params:
            input = (prediction, lengths)
            prediction:  (batch_size, pad_length) Tensor of predicted values for the batch
            target:  (batch_size, pad_length) Tensor of target values for the batch
            lengths: (batch_size)  Tensor of valid lengths for each sequence in the batch

        Returns:
            loss:  (Float) loss over the current batch
        """
        shape_debug = False
        prediction, lengths = input
        pad_length = prediction.size()[1]
        if shape_debug: logging.info("Loss Predictions: {}".format(prediction.size()))
        if shape_debug: logging.info("Loss Targets: {}".format(target.size()))
        if shape_debug: logging.info("Loss Lengths: {}".format(lengths.size()))
        if shape_debug: logging.info("Loss Pad Length: {}".format(pad_length))
        all_cross_entropies = torch.mul(prediction[:,:,0], target[:,:,0])
        if shape_debug: logging.info("All CE values (pos class only, drop neg-class): {}".format(all_cross_entropies.size()))

        # Mask and compute SE over valid subsequences for all sequences in the batch
        # Sets squared-error to zero past the valid lengths
        length_mask = length_to_mask(lengths.view(-1), pad_length, dtype=torch.float32).view(-1, pad_length)
        if shape_debug: logging.info("Length Masks: {}".format(length_mask.size()))
        # length_mask = length_mask.unsqueeze(2).repeat(1, 1, 2)
        # if shape_debug: logging.info("Tiled Length Masks: {}".format(length_mask.size()))
        masked_cross_entropies = torch.mul(all_cross_entropies, length_mask)
        
        # Average the errors over sequence length independently for each sample in the batch
        sample_cross_entropies = torch.div(torch.sum(masked_cross_entropies, dim=1, keepdim=False),
                                           lengths.float())
        
        # Averaged over all samples in the batch so loss is invariant of batch size
        loss = -torch.mean(sample_cross_entropies)

        return loss
    