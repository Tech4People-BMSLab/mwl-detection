"""-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-07-17 15:42:27
@Last Modified by:   Tenzing Dolmans
@Last Modified time: 2020-08-17 11:03:28
@Description: Takes hierarchically organised CSV files
and writes a TFRecord file. Parsing is done with "tfrecord_reader.py"
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import list_files_recursively  # noqa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # noqa


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte.
    Retrieved from:
    https://www.tensorflow.org/tutorials/load_data/tfrecord"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # Won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    "Returns a float_list from a float / double."
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    "Returns an int64_list from a bool / enum / int / uint."
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class TFRecordCreator:
    """TFRecord file writer that loops over participant folders.
    Works with files that are created using the chunkify.py script.
    Params:
        tfrecord_file: File you want to write.
        weights: CSV file that contains the indicated puzzle difficulties
                 for all participants.
        p_data: Folder that contains one folder per participant,
                each of which contain respective label files and XDF."""
    def __init__(self, tfrecord_file, weights, p_data):
        self.writer = tf.io.TFRecordWriter(tfrecord_file)
        self.tfrecord_file = tfrecord_file
        self.weights = pd.read_csv(weights)
        self.p_data = p_data

    def add_to_tfrecords(self, partno, ppg_shape, gsr_shape, nirs_height,
                         nirs_width, et_height, et_width, ppg, gsr, nirs,
                         et, label):
        "Call that adds samples to the TFRecord file."
        example = tf.train.Example(features=tf.train.Features(feature={
            'partno': _int64_feature(partno),
            # 'age': _int64_feature(age),
            # 'gender': _bytes_feature(gender.encode('utf-8')),
            'ppg_shape': _int64_feature(ppg_shape),
            'gsr_shape': _int64_feature(gsr_shape),
            'nirs_height': _int64_feature(nirs_height),
            'nirs_width': _int64_feature(nirs_width),
            'et_height': _int64_feature(et_height),
            'et_width': _int64_feature(et_width),
            'ppg': _bytes_feature(ppg),
            'gsr': _bytes_feature(gsr),
            'nirs': _bytes_feature(nirs),
            'et': _bytes_feature(et),
            'label': _bytes_feature(label)}))
        self.writer.write(example.SerializeToString())

    def save_to_tfrecord(self):
        """Grabs data from hierarchical structure and unpacks all values.
        Calls add_to_tfrecords to add gathered data to a TFRecord file as
        as single sample."""
        for i, participant in enumerate(self.weights.partno):
            print("Working on ", participant)

            # Select the correct puzzle difficulties for current participant
            part_weights = self.weights[self.weights.partno ==
                                        int(participant)]
            part_weights = part_weights.iloc[0][1:6]

            # Define which folder we are working in based on participant
            data_path = self.p_data + 'P{:02d}'.format(i+1)
            marker_df = pd.read_csv(data_path + '\\labels_{}.csv'
                                    .format(participant))
            marker_df.condition = marker_df.condition.replace(
                ['vlow', 'low', 'mid', 'high', 'vhigh'],
                [part_weights['vlow'], part_weights['low'],
                 part_weights['mid'], part_weights['high'],
                 part_weights['vhigh']])

            # Select relevant data for every entry in marker_df
            for i in range(len(marker_df)):
                ppg = np.genfromtxt(data_path + '\\{}_PPG.csv'
                                    .format(i), delimiter=',')
                gsr = np.genfromtxt(data_path + '\\{}_GSR.csv'
                                    .format(i), delimiter=',')
                nirs = np.genfromtxt(data_path + '\\{}_NIRS.csv'
                                     .format(i), delimiter=',')
                et = np.genfromtxt(data_path + '\\{}_ET.csv'
                                   .format(i), delimiter=',')
                ppg_shape = ppg.shape[0]
                gsr_shape = gsr.shape[0]
                nirs_height, nirs_width = nirs.shape
                et_height, et_width = et.shape

                # Some formatting to comply with tf.Train.Feature functions
                ppg = list(ppg.flatten())
                ppg = tf.io.serialize_tensor(ppg)
                gsr = list(gsr.flatten())
                gsr = tf.io.serialize_tensor(gsr)
                nirs = list(nirs.flatten())
                nirs = tf.io.serialize_tensor(nirs)
                et = list(et.flatten())
                et = tf.io.serialize_tensor(et)
                label = [marker_df.condition.iloc[i]]

                # Sanity check on the labels
                if label[0] < 0:
                    print("Went wrong at: {}, {}!".format(participant, i))
                label = tf.io.serialize_tensor(label)
                self.add_to_tfrecords(participant, ppg_shape, gsr_shape,
                                      nirs_height, nirs_width, et_height,
                                      et_width, ppg, gsr, nirs, et, label)


if __name__ == "__main__":
    p_data = 'Path\\To\\MWL-Detection\\Data\\p_data\\'
    weights = 'Path\\To\\MWL-Detection\\Data\\part_weights.csv'
    files, _ = list_files_recursively(p_data, extension='.xdf')
    t = TFRecordCreator('OutputName.tfrecord', weights, p_data)
    t.save_to_tfrecord()
