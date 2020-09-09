"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-07-17 11:58:10
@Last Modified by:   Tenzing Dolmans
@Last Modified time: 2020-08-05 14:06:05
@Description: Reads and parses TFRecord data files.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa
import tensorflow as tf
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


def tfrecord_dataset(batch_size, dataset):
    """Function that is called to read and parse TFRecord files."""
    dataset = dataset.map(dataset_parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def dataset_parser(tfrecord):
    """Parses datasets that were made using the
    TFRecordCreator from tfrecord_writer.py"""
    features = {
        'partno': tf.io.FixedLenFeature([], tf.int64),
        'ppg_shape': tf.io.FixedLenFeature([], tf.int64),
        'gsr_shape': tf.io.FixedLenFeature([], tf.int64),
        'nirs_height': tf.io.FixedLenFeature([], tf.int64),
        'nirs_width': tf.io.FixedLenFeature([], tf.int64),
        'et_height': tf.io.FixedLenFeature([], tf.int64),
        'et_width': tf.io.FixedLenFeature([], tf.int64),

        'ppg': tf.io.FixedLenFeature([], tf.string),
        'gsr': tf.io.FixedLenFeature([], tf.string),
        'nirs': tf.io.FixedLenFeature([], tf.string),
        'et': tf.io.FixedLenFeature([], tf.string),
        'label':  tf.io.FixedLenFeature([], tf.string),
    }

    sample = tf.io.parse_single_example(serialized=tfrecord, features=features)
    ppg = tf.io.parse_tensor(sample['ppg'], out_type=tf.float64)
    ppg = tf.dtypes.cast(ppg, tf.float32)
    gsr = tf.io.parse_tensor(sample['gsr'], out_type=tf.float64)
    gsr = tf.dtypes.cast(gsr, tf.float32)
    nirs = tf.io.parse_tensor(sample['nirs'], out_type=tf.float64)
    nirs = tf.reshape(nirs, [sample['nirs_height'], sample['nirs_width']])
    nirs = tf.dtypes.cast(nirs, tf.float32)
    et = tf.io.parse_tensor(sample['et'], out_type=tf.float64)
    et = tf.reshape(et, [sample['et_height'], sample['et_width']])
    et = tf.dtypes.cast(et, tf.float32)
    label = tf.io.parse_tensor(sample['label'], out_type=tf.float64)
    label = tf.dtypes.cast(label, tf.float32)
    partno = sample['partno']
    return ppg, gsr, nirs, et, label, partno


if __name__ == "__main__":
    file = 'Filename.tfrecord'
    raw_dataset = tf.data.TFRecordDataset(file)
    dataset = tfrecord_dataset(8, raw_dataset)
    for batch in dataset:
        ppg, gsr, nirs, et, label, partno = batch
