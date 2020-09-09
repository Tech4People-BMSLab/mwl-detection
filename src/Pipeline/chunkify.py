"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-05-07 12:23:15
@Last Modified by:   Tenzing Dolmans
@Description: Contains functions and __main__ loop to convert
listed files to a .CSV based dataset.
"""
import logging
import os
import time
import numpy as np
from itertools import compress

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import explore_markers, is_disjoint, list_files  # noqa
from xdf_import import read_xdf  # noqa


def create_chunk(input_dict, out_folder,
                 epoch_len=8, data_column='time_series',
                 time_column='time_stamps'):
    """
    Purpose:
        Select and epoch small chunks of data for all devices
        in a datastream around indicated markers of interest.
        The selection is done using a dict from explore_markers().
    Args:
        input_dict : Dictionary as a result of read_xdf().
        out_folder:  Name of the folder(s) to output files to.
        epoch_len :   Length of data selection, value in seconds.
                     Selection is done BEFORE the time of the marker.
        data_column: Column in input_dict to extract data from. Options are
                     currently 'time_series' and 'normalised'.
        time_column: Column in input_dict to base epoch timing on.
    """
    # Keeping track of how long the function takes
    start = time.process_time()
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    soi = []
    names = input_dict.name

    # Check which stream is the marker stream, and which are data
    # Datastreams are added to the 'stream of interest' (soi) mask
    for i, st in enumerate(names):
        if st == ['ZebraMarkerStream']:
            print('Stream {} is the marker stream: {}'.format(i, st))
        else:
            soi.append(i)
            print('Stream {} {} is a data stream'.format(i, st,))

    entries, _, timestamps = explore_markers(input_dict)
    device_time = [np.asarray(array) for array in
                   input_dict[time_column].iloc[soi]]

    # Find nearest timestamps for each marker in every device:
    indices = [[(np.abs(array - entry)).argmin() for array in device_time]
               for entry in timestamps]

    """Uncomment below when making specific selections of data
    for a single participant."""
    # entries = entries[timestamps > 176103]

    # Remove markers that point to the same data more than once
    entries_mask, indices = is_disjoint(indices)
    entries = entries[entries_mask]

    # Only select "answers" in markers
    indices_mask = [value == 'correct' or value == 'incorrect'
                    for value in entries.status]
    indices = list(compress(indices, indices_mask))
    entries = entries[indices_mask]
    entries = entries.reset_index()

    # Save the selected markers as a file
    entries.to_csv(out_folder + '\\labels_{}.csv'
                   .format(entries.partno.iloc[0]))

    # Get the sampling rate and data type of each device
    sr = [value[0] for value in input_dict['sampling_rate'].iloc[soi]]
    types = [value[0] for value in input_dict['type'].iloc[soi]]

    # Select the data for every device in soi
    device_data = [np.asarray(array) for array in
                   input_dict[data_column].iloc[soi]]

    # Loop over all the indices that made it through selection
    limit = len(entries)
    for i, marker_number in enumerate(indices):
        if i >= limit:
            print("Something went wrong, more indices than markers.")
        else:
            # Loop over devices for all indices and output files
            for j, device_stamp in enumerate(marker_number):
                # TODO: different selections for different devices
                begin = device_stamp - int(sr[j]) * epoch_len
                data_selection = device_data[j][begin:device_stamp]
                filename = out_folder + '\\{}_{}.csv'.format(i, types[j])
                np.savetxt(filename, data_selection, delimiter=",")
                # print('Created file marker_{}_{}.csv'.format(i, types[j]))
    print("Done creating files! It took {:02.02f} seconds."
          .format(time.process_time()-start))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for more output
    all_data = 'Path/to/folder/with/all/XDF/files'
    p_data = 'Path/to/output/folder'
    files = list_files(all_data)
    for ix, file in enumerate(files):
        print("Doing: ", file[-7:-3])  # Select Participant Number from file
        data = read_xdf(file)
        create_chunk(data, out_folder=p_data + '\\' + file[-7:-3])

    # Uncomment below for participant-specific file creation
    # file = files[15]
    # data = read_xdf(file)
    # create_chunk(data, out_folder=p_data + '\\FolderName')
