"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-05-12 09:10:56
@Last Modified by:   Tenzing Dolmans
@Description: Contains various utility functions that are used
throughout the rest of the project.
"""
import os
import glob
import matplotlib.pyplot as plt
from itertools import compress
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa


def list_files(directory):
    "List all files in the dataset directory."
    files = []
    for filename in os.listdir(directory):
        # if filename.endswith(".csv"):
        single = os.path.join(directory, filename)
        files.append(single)
    return files


def list_files_recursively(directory, extension='.csv'):
    "List all files in the dataset directory recursively."
    path_list = [os.path.relpath(f, directory) for f in
                 glob.iglob(directory + f'**/*{extension}', recursive=True)]
    no_files = len(path_list)
    return path_list, no_files


def build_tensors(files):
    """DEPRECATED FUNCTION
    Takes files from list_files() and converts to tensors for training.
    DEVICES ARE HARD-CODED NOW, THIS REQUIRES MANUAL ATTENTION. ++
    """
    ppg = [pd.read_csv(file, header=None).to_numpy() for file in files
           if 'PPG' in file]
    gsr = [pd.read_csv(file, header=None).to_numpy() for file in files
           if 'GSR' in file]
    et = [pd.read_csv(file, header=None).to_numpy() for file in files
          if 'ET' in file]
    nirs = [pd.read_csv(file, header=None).to_numpy() for file in files
            if 'NIRS' in file]
    ppg = tf.keras.utils.normalize(ppg)
    gsr = tf.keras.utils.normalize(gsr)
    et = tf.keras.utils.normalize(et, axis=1, order=2)
    nirs = tf.keras.utils.normalize(nirs, axis=1, order=2)
    tensor_ppg = tf.convert_to_tensor(ppg[:, :, 0])
    tensor_gsr = tf.convert_to_tensor(gsr[:, :, 0])
    tensor_et = tf.convert_to_tensor(et)
    tensor_nirs = tf.convert_to_tensor(nirs)
    return [tensor_ppg, tensor_gsr, tensor_et, tensor_nirs]


def build_datasets(tensors, targets):
    """DEPRECATED FUNCTION
    Makes TF datasets from tensors.
    """
    ppg, gsr, et, nirs = tensors
    dataset_ppg = tf.data.Dataset.from_tensor_slices((ppg, targets))
    dataset_gsr = tf.data.Dataset.from_tensor_slices((gsr, targets))
    dataset_et = tf.data.Dataset.from_tensor_slices((et, targets))
    dataset_nirs = tf.data.Dataset.from_tensor_slices((nirs, targets))
    return dataset_ppg, dataset_gsr, dataset_et, dataset_nirs


def get_part_weights(weights, marker_df):
    """DEPRECATED FUNCTION
    Finds the right weights based on partno.
    Requires weight file and marker_df for that participant.
    """
    partno = pd.Categorical(marker_df.partno)
    current = weights[weights.partno == int(partno[0])]
    return partno, current.iloc[0][1:6]


def explore_sync(input_dict, verbose=False):
    """
    This function explores the desynchronisation in time_stamps
    from indicated devices in your streams. Plots if verbose
    is set to True. Visually, it works best when comparing streams
    that have similar sampling rates. The difference between the
    horizontal lines is the desynchronisation.
    """
    highest = [np.max(entry) for entry in input_dict['duration']]
    names = [entry for entry in input_dict['type']]

    # Remove any marker streams
    for ix, entry in enumerate(names):
        if entry == ['Markers']:
            del highest[ix]
            del names[ix]
    top = np.max(highest)
    differences = [top - high for high in highest]
    drift = np.max(differences)

    # Drift per minute
    dpm = drift / (top / 60)
    percentage = (dpm / 60) * 100
    longest = names[highest.index(top)]
    shortest = names[differences.index(drift)]
    if verbose:
        print('The biggest calculated drift is: \n{:04.4f} seconds.'
              .format(drift))
        print('The longest recording is {}, the shortest is {}'
              .format(longest, shortest))
        print('\nThis results in {:04.4f} seconds per minute of recording,'
              ' or {:02.4f} percent.'
              .format(np.max(dpm), np.max(percentage)))

        # Plot a line for each stream, label accordingly
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(input_dict))))
        for ix, stream in enumerate(input_dict['duration']):
            c = next(color)
            plt.plot(stream, color=c, label=input_dict['type'].iloc[ix])
            # Add a horizontal line at the final timestamp
            plt.axhline(y=np.max(stream), color=c)
        plt.xlabel('# Timestamps')
        plt.ylabel('Stream duration (s)')
        plt.legend(loc=2)
        plt.show()
    return drift


def explore_markers(all_data, plot=False):
    """Inspect the ZebraMarkerStream stream in an XDF file and
    calculate some statistics about all markers in said stream.
    Plots time between answers if plot is True."""
    if len(all_data) > 1:
        markers = [
            [all_data['time_series'][stream], all_data['time_stamps'][stream]]
            for stream in range(len(all_data))
            if all_data['name'][stream] == ['ZebraMarkerStream']]
    else:
        markers = all_data.time_series

    df = pd.DataFrame(markers[0][0], columns=(
        'partno', 'timestamp', 'type', 'id',
        'status', 'puzzle', 'condition'))
    timestamps = markers[0][1]
    answers = df['status'].value_counts(dropna=False)

    # Select which mask you want to select the markers with
    mask = df['timestamp'][df.status == 'correct']
    times = [int(df['timestamp'][key]) for key in mask.keys()]
    between = [times[current] - times[current - 1]
               for current in range(len(times))]
    # Remove negative values that are meaningless
    del(between[0:2])

    time_between = np.mean(between)

    # Compute a ratio of correctness
    if len(answers) > 2:
        ratio_correct = answers.correct/(answers.incorrect + answers.correct)
    else:
        ratio_correct = 1
    stats = [times, between, time_between, ratio_correct]

    if plot:
        print('Mean Time between answers: {:04.02f}s\nRatio correct: {:02.02f}'
              .format(time_between/1000, ratio_correct))
        plt.plot(between)
        plt.xlabel('Marker number')
        plt.ylabel('Time between markers (ms)')
        plt.show()
    return df, stats, timestamps


def is_disjoint(in_list):
    "Remove all entries from a list that are not disjoint."
    mask = [set(in_list[i]).isdisjoint(set(in_list[i-1]))
            for i in range(len(in_list))]
    return mask, list(compress(in_list, mask))
