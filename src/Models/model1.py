"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-05-07 12:20:58
@Last Modified by:   Tenzing Dolmans
@Description: Deprecated version of model that makes use of
multiple bases. Not used to create any of the published results.
"""
from utils import list_files, build_tensors, build_targets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa
from tensorflow.keras import layers # noqa


class BaseLayer(layers.Layer):
    def __init__(self,
                 filters=64,
                 kernel_size=3,
                 use_bias=True,
                 strides=1,
                 padding='valid',
                 activation='relu',
                 dropout_rate=0.2,
                 name=''):
        super(BaseLayer, self).__init__()
        self.bn = layers.BatchNormalization()
        self.C = layers.Conv1D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               use_bias=use_bias,
                               padding=padding,
                               activation=activation,
                               name=name)
        self.mp = layers.MaxPool1D(pool_size=2)
        self.drop = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=False):
        x = self.bn(inputs, training=training)
        x = self.C(x)
        x = self.mp(x)
        x = self.drop(x, training=training)
        return x


class ShimmerBase(tf.keras.Model):
    def __init__(self,
                 activation='relu',
                 units=64):
        super(ShimmerBase, self).__init__()
        self.C1 = BaseLayer(name='S_C_1')
        self.C2 = BaseLayer(name='S_C_2',
                            filters=128)
        self.L1 = layers.LSTM(units=units,
                              name="S_L_2",
                              return_sequences=True)
        self.L2 = layers.LSTM(units=128,
                              name="S_L_1",
                              return_sequences=True)

    def call(self, inputs, training=False):
        x = self.C1(inputs, training=training)
        x = self.C2(x, training=training)
        x = self.L1(x)
        x = self.L2(x)
        return x


class TobiiBase(tf.keras.Model):
    def __init__(self,
                 activation='relu',
                 units=64,
                 dropout_rate=0.3):
        super(TobiiBase, self).__init__()
        self.C1 = BaseLayer(name="T_C_1")
        self.C2 = BaseLayer(name="T_C_2",
                            filters=128)
        self.C3 = BaseLayer(name="T_C_3",
                            filters=128,
                            kernel_size=3)
        self.L1 = layers.LSTM(units=units,
                              name="T_L_1",
                              return_sequences=True)
        self.L2 = layers.LSTM(units=128,
                              name="T_L_2",
                              dropout=dropout_rate,
                              return_sequences=True)

    def call(self, inputs, training=False):
        x = self.C1(inputs, training=training)
        x = self.C2(x, training=training)
        x = self.C3(x, training=training)
        x = self.L1(x)
        x = self.L2(x)
        return x


class BriteBase(tf.keras.Model):
    def __init__(self, activation='relu', dropout_rate=0.3, units=128):
        super(BriteBase, self).__init__()
        self.C1 = BaseLayer(strides=2,
                            name="B_C_1",)
        self.C2 = BaseLayer(filters=units,
                            name="B_C_2")
        self.C3 = BaseLayer(filters=256,
                            name="B_C_3")
        self.drop = layers.Dropout(rate=dropout_rate)
        self.dense = layers.Dense(units=units,
                                  activation=activation)
        self.L1 = layers.LSTM(units=64,
                              name="B_L_1",
                              return_sequences=True)
        self.L2 = layers.LSTM(units=units,
                              name="B_L_2",
                              return_sequences=True)

    def call(self, inputs, training=False):
        x = self.C1(inputs, training=training)
        x = self.C2(x, training=training)
        x = self.C3(x, training=training)
        x = self.drop(x)
        x = self.dense(x)
        x = self.L1(x)
        x = self.L2(x)
        return x


class HeadClass(tf.keras.Model):
    def __init__(self, dropout_rate, filters=128, activation='relu'):
        super(HeadClass, self).__init__()
        self.shimmer = ShimmerBase()
        self.tobii = TobiiBase()
        self.brite = BriteBase()
        self.flat = layers.Flatten()
        self.reshape = layers.Reshape(target_shape=(32, 32, 30))
        self.fuse = layers.Conv2D(filters=filters,
                                  kernel_size=5,
                                  name="H_F")
        self.C1 = layers.Conv2D(filters=256,
                                kernel_size=3,
                                name="H_C_1")
        self.drop = layers.Dropout(rate=dropout_rate)
        self.C2 = layers.Conv2D(filters=256,
                                kernel_size=3,
                                name="H_C_2")
        self.D1 = layers.Dense(units=256,
                               activation=activation,
                               name="H_D_1")
        self.D2 = layers.Dense(units=filters,
                               activation=activation,
                               name="H_D_2")
        self.out = layers.Dense(units=5,
                                activation='sigmoid')

    def call(self, inputs, training=False):
        shim, et, nirs = inputs
        shimmer_out = self.shimmer(shim, training=training)
        tobii_out = self.tobii(et, training=training)
        brite_out = self.brite(nirs, training=training)
        # print('ShimShape:{}'.format(shimmer_out.shape))
        # print('TobiiShape:{}'.format(tobii_out.shape))
        # print('BriteShape:{}'.format(brite_out.shape))
        x = self.flat(shimmer_out)
        y = self.flat(tobii_out)
        z = self.flat(brite_out)
        concat = tf.concat([x, y, z], axis=1)
        # print('ConcatShape:{}'.format(concat.shape))
        x = self.reshape(concat)
        x = self.fuse(x)
        x = self.C1(x)
        x = self.drop(x)
        x = self.C2(x)
        x = self.D1(x)
        x = self.D2(x)
        out = self.out(x)
        return out


def train_from_tensors(dataset_directory, epochs=4,
                       batch_size=1, dropout_rate=0.3):
    files = list_files(dataset_directory)
    all_tensors = build_tensors(files)
    targets = build_targets(files)
    model = HeadClass(dropout_rate=dropout_rate)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # ckpt = tf.train.Checkpoint()
    model.fit(all_tensors, targets, validation_split=0.2,
              epochs=epochs, batch_size=batch_size)
    model.summary()
    # model.save_weights('Checks/check')


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    dataset_directory = "C:/Users/DolmansTC/mwl-detection/XDF/Data/Dataset"
    train_from_tensors(dataset_directory)
    print("Banana")
