""" Utility functions for the Waveform Statistics project
"""

import csv
import json
import logging
import os
import shutil
import collections
import argparse
import numpy as np
import pandas as pd
import torch


def date_lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.apply(lambda v: dates[v])


def count_csv_rows(filepath, sep=',', header=True):
    """ Count the number of rows in a csv file
    """
    with open(filepath) as f:
        row_count = sum(1 for line in f)
    # fileObject = csv.reader(filepath, delimiter=sep)
    # row_count = sum(1 for row in fileObject)
    row_count -= int(header)
    return row_count


def compute_lookahead_seconds(params):
    """ Compute how many seconds the convolutional network must lookahead
        based on strides and pools
    """
    depth = len(params.conv_strides)
    layer_total_widths = []
    for l in range(depth):
        layer_total_widths.append(params.conv_kernel_sizes[l] * params.conv_strides[l] * 2**params.maxpool_layers[l])
    
    total_width = np.prod(layer_total_widths)
    total_lookahead = (total_width / 2.) / 125.  # in seconds
    logging.info("Successive layer widths: {}\nTotal width: {}\nLookahead: {} sec".format(
        layer_total_widths, total_width, total_lookahead))

    if params.lookahead_limit > 0:
        if params.lookahead_limit >= total_lookahead:
            logging.info("Total lookahead of {:.2f} under limit of {} seconds".format(
                total_lookahead, params.lookahead_limit))
        else:
            logging.info("Total lookahead of {:.2f} exceeds limit of {} seconds".format(
                total_lookahead, params.lookahead_limit))


def rolling_window_average(a, window=3, keep_dim=True) :
    """ Compute rolling window average of numpy array
        Pad to leading edge to keep dimensions
    """
    if keep_dim:
        a = np.pad(a, (window-1, 0), mode='edge')
        # print("\npadded: ({}):  {}".format(a.shape, a))
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def runs_of_ones_array(bits):
    """ Source: 
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    """
    # Remove all data not 0 or 1
    bits = bits[np.in1d(bits, np.array([0,1]))]

    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_ends - run_starts

def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

def convolution_out_length(in_length, kernel_size, stride=1, padding=1, dilation=1):
    """ https://pytorch.org/docs/stable/nn.html#conv1d
    Computes the output length dimension for the pytorch.nn.conv1d layer
    """
    return int((in_length + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)

def compute_conv_padding(kernel_size, stride, dilation=1):
    """ Compute the proper padding level so that out_length = in_length/stride
    Note:  if dilation=1, kernel_size must be even

    Need:  -stride <= 2×padding−dilation×(kernel_size−1)−1 < 0 
    """
    assert (dilation % 2 == 0) or (kernel_size % 2 == 1), "Kernel size {} must be even for odd dilation {}".format(kernel_size, dilation)

    padding = int(np.ceil((dilation*(kernel_size-1) - stride + 1) / 2))
    # assert padding == int(padding), "Error: Padding {} must be integer".format(padding)
    return padding

def compute_deconv_padding(kernel_size, stride):
    """ Compute the proper padding level so that out_length = in_length/stride
    Note:  if dilation=1, kernel_size must be even

    Need:  -stride <= 2×padding−dilation×(kernel_size−1)−1 < 0 
    """
    output_padding = int(((kernel_size - stride) % 2 == 1))
    padding = int((kernel_size - stride + output_padding) / 2)
    return padding, output_padding

def check_numpy_array_file(filename):
    print("Checking numpy array at {}".format(filename))
    data = np.load(filename)
    print("Shape: {}".format(data.shape))
    print("Type: {}".format(np.result_type(data)))
    print("Up to 5:\n{}".format(data[:5]))


def write_df_to_table(df, filename, precision = '%.2f'):
    """ Convert a pandas dataframe to latex table and write to file
    """
    float_formatter=lambda x: precision % x

    table = df.to_latex(float_format=float_formatter)
    logging.debug("Writing table to {}:\n{}".format(filename, table))
    with open(filename, 'w') as tfile:
        tfile.write(table)


def ensure_directory(dirs):
    """ Create directory/ies if it/they doesn't already exist
    """
    # Listify if necessary
    if not isinstance(dirs, (list, tuple)):
        dirs = [dirs]

    for directory in dirs:
        if not os.path.exists(directory):
            logging.debug("Creating new directory {}".format(directory))
            os.makedirs(directory)


def str2bool(v):
    """ Used for boolean options in the argparse interpretation
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
def set_logger(log_path, level=logging.INFO):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    logdir = os.path.split(log_path)[0]
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Params():
    """Class that loads hyperparameters from a json file or dictionary.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, inputs):
        if type(inputs) is dict:
            # Load params from already defined dict
            self.__dict__.update(inputs)
        else:
            # Load parameters from json file
            with open(inputs) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        # Save parameters to json file
        import utils
        import pathlib
        utils.ensure_directory(os.path.dirname(json_path))
        for k,v in self.__dict__.items():
            if type(v) is pathlib.PosixPath:
                self.__dict__[k] = str(v)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, inputs):
        # Load parameters from json file
        if type(inputs) is dict:
            self.__dict__.update(inputs)
        else:
            with open(inputs) as f:
                params = json.load(f)
                self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        if not np.isnan(val) and (val >= 0):
            self.total += val
            self.steps += 1
    
    def __call__(self):
        if self.steps == 0:
            return 0
        else:
            return self.total/float(self.steps)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_dict_from_json(json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as jfile:
        jstr = jfile.read()
        d = json.loads(jstr)
    return d


def save_list_to_json(l, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        l = [float(v) for v in l]
        json.dump(l, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    # else:
    #     print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_dict(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4, separators=(',', ': '))

def load_dict(filename):
    with open(filename) as jfile:
        jdict = json.load(jfile)
    return jdict


if __name__ == '__main__':
    # Run unit tests
    a = np.arange(10)
    print("a ({}):  {}".format(a.shape, a))
    for w in range(1, 11):
        m = rolling_window_average(a, w, keep_dim=True)
        print("w={}\nm ({}):  {}".format(w, m.shape, m))