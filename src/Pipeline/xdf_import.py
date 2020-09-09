"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-05-07 12:23:55
@Last Modified by:   Tenzing Dolmans
@Description: Generic LabRecorder XDF to dataframe converter.
"""
import pyxdf
import pandas as pd
import numpy as np
import logging
import os


def read_xdf(file):
    """
    Takes an XDF file and outputs a DataFrame with:
        info: name, srate, and channel count
        time_stamps: timestamps for each data point
        time_series: all data points
        duration: total duration that data points span
    """
    streams, _ = pyxdf.load_xdf(file)

    # Sanity check on the number of samples
    streams = [st for st in streams if (
        len(st['time_stamps']) > 10)]
    df = pd.DataFrame(columns=(
        'name', 'type', 'sampling_rate', 'channel_count',
        'time_stamps', 'time_series', 'duration'))

    for ix, st in enumerate(streams):
        np.nan_to_num(st)
        df = df.append({'name': st['info']['name']}, ignore_index=True)
        df['type'].iloc[ix] = st['info']['type']
        df['sampling_rate'].iloc[ix] = st['info']['nominal_srate']
        df['channel_count'].iloc[ix] = st['info']['channel_count']
        df['time_stamps'].iloc[ix] = st['time_stamps']
        df['time_series'].iloc[ix] = st['time_series']
        df['time_series'].iloc[ix] = np.nan_to_num(df['time_series'].iloc[ix])
        df['duration'].iloc[ix] = st['time_stamps'] - st['time_stamps'][0]
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Logging.DEBUG for more output
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_file = 'Path/to/file'  # noqa
    all_data = read_xdf(data_file)
