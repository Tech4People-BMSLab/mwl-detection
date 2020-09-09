"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-21-07 12:20:58
@Last Modified by:   Tenzing Dolmans
@Description: Small version of "literature_base.py",
meaning that all layers have half the units/filters.
This file contains bases for PPG, GSR, ET, and fNIRS,
as well as one head model. The bases are based on what is commonly
done for the respective modality in literature. The head model uses
some dense and convolutional layers.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa
from tensorflow.keras import layers # noqa


class PPGBase(tf.keras.Model):
    """Base for the PPG Modality. Expected input data is 1D.
    Model adapted from Sun, Hong, and Ren (2019),
    Hybrid spatiotemporal models."""
    def __init__(self,
                 activation='relu',
                 filters=64,
                 units=128):
        super(PPGBase, self).__init__()
        self.C1 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="P_C_1")
        self.BN1 = layers.BatchNormalization(name="P_N_1")
        self.C2 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="P_C_2")
        self.BN2 = layers.BatchNormalization(name="P_N_2")
        self.MP = layers.MaxPool1D(name="P_P_1")
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="P_D_1")

    def call(self, inputs, training=False):
        x = tf.reshape(inputs, shape=[-1, 256, 8])
        x = self.C1(x)
        x = self.BN1(x, training=training)
        x = self.C2(x)
        x = self.BN2(x, training=training)
        x = self.MP(x)
        # Flatten and convert to dense representation
        x = self.flat(x)
        x = self.D1(x)
        return x


class GSRBase(tf.keras.Model):
    """Base for the GSR Modality. Expected input data is 1D.
    Model adapted from Biswas et al. (2019), CorNET."""
    def __init__(self,
                 activation='relu',
                 filters=64,
                 units=128):
        super(GSRBase, self).__init__()
        self.C1 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="G_C_1")
        self.BN1 = layers.BatchNormalization(name="G_N_1")
        self.C2 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="G_C_2")
        self.BN2 = layers.BatchNormalization(name="G_N_2")
        self.MP = layers.MaxPool1D(name="G_P_1")
        self.L1 = layers.LSTM(units=units,
                              name="G_L_1",
                              return_sequences=True)
        self.L2 = layers.LSTM(units=units,
                              name="G_L_2",
                              return_sequences=True)
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="G_D_1")

    def call(self, inputs, training=False):
        x = tf.reshape(inputs, shape=[-1, 256, 8])
        x = self.C1(x)
        x = self.BN1(x, training=training)
        x = self.C2(x)
        x = self.BN2(x, training=training)
        x = self.MP(x)
        x = self.L1(x, training=training)
        x = self.L2(x, training=training)
        # Flatten and convert to dense representation
        x = self.flat(x)
        x = self.D1(x)
        return x


class BriteBase(tf.keras.Model):
    """Base for the GSR Modality. Expected input data is 2D.
    Model adapted from Dargazany, Abtahi, Mankodiya (2019),
    arXiv:1907.09523. """
    def __init__(self,
                 activation='relu',
                 filters=256,
                 units=1024):
        super(BriteBase, self).__init__()
        self.C1 = layers.Conv1D(filters=filters,
                                kernel_size=5,
                                activation=activation,
                                name="B_C_1")
        self.BN1 = layers.BatchNormalization(name="B_N_1")
        self.C2 = layers.Conv1D(filters=filters,
                                kernel_size=5,
                                activation=activation,
                                name="B_C_2")
        self.BN2 = layers.BatchNormalization(name="B_N_2")
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_1")
        self.BN3 = layers.BatchNormalization(name="B_N_3")
        self.D2 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_2")
        self.BN4 = layers.BatchNormalization(name="B_N_4")
        self.D3 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_3")

    def call(self, inputs, alpha=0.1, training=False):
        x = self.C1(inputs)
        x = self.BN1(x, training=training)
        x = self.C2(x)
        x = self.BN2(x, training=training)
        x = self.flat(x)
        x = self.D1(x)
        x = self.BN3(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D2(x)
        x = self.BN4(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D3(x)
        return x


class TobiiBase(tf.keras.Model):
    """Base for the ET Modality. Expected input data is 2D.
    Model adapted from:
    Louedec et al. (2019), Chess player attention prediction;
    Krafka et al. (2016), Eye tracking for everyone."""
    def __init__(self,
                 activation='relu',
                 filters=128,
                 units=512):
        super(TobiiBase, self).__init__()
        self.C1 = layers.Conv1D(filters=filters,
                                kernel_size=4,
                                activation=activation,
                                name="T_C_1")
        self.BN1 = layers.BatchNormalization(name="T_N_1")
        self.C2 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="T_C_2")
        self.MP = layers.MaxPool1D(pool_size=4,
                                   name="T_P_1")
        self.C3 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="T_C_3")
        self.BN2 = layers.BatchNormalization(name="T_N_2")
        self.C4 = layers.Conv1D(filters=filters,
                                kernel_size=2,
                                activation=activation,
                                name="T_C_4")
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="T_D_1")

    def call(self, inputs, training=False):
        x = self.C1(inputs)
        x = self.BN1(x, training=training)
        x = self.C2(x)
        x = self.MP(x)
        x = self.C3(x)
        x = self.BN2(x, training=training)
        x = self.C4(x)
        x = self.MP(x)
        # Flatten and convert to dense representation
        x = self.flat(x)
        x = self.D1(x)
        return x


class HeadClass(tf.keras.Model):
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    def __init__(self, dropout_rate=0.2,
                 filters=256, units=1024):
        super(HeadClass, self).__init__()
        self.ppg = PPGBase()
        self.gsr = GSRBase()
        self.brite = BriteBase()
        self.tobii = TobiiBase()
        self.D1 = layers.Dense(units=1792,  # Number of units of all bases.
                               name="H_D_1")
        self.BN1 = layers.BatchNormalization(name="H_N_1")
        self.D2 = layers.Dense(units=units*2,
                               name="H_D_2")
        self.reshape = layers.Reshape(target_shape=(32, 16, 8))
        self.C1 = layers.Conv2D(filters=filters,
                                kernel_size=3,
                                padding='same',
                                name="H_C_1")
        self.C2 = layers.Conv2D(filters=filters,
                                kernel_size=3,
                                padding='same',
                                name="H_C_2")
        self.BN2 = layers.BatchNormalization(name="H_N_2")
        self.C3 = layers.Conv2D(filters=filters/2,
                                kernel_size=3,
                                padding='same',
                                name="H_C_3")
        self.flat = layers.Flatten()
        self.D3 = layers.Dense(units=units/4,
                               name="H_D_1")
        self.BN3 = layers.BatchNormalization(name="H_N_3")
        self.D4 = layers.Dense(units=units/8,
                               name="H_D_2")
        self.out = layers.Dense(units=1,
                                activation='sigmoid')

    def call(self, ppg, gsr, nirs, et, alpha=0.1, training=False):
        ppg_out = self.ppg(ppg, training=training)
        gsr_out = self.gsr(gsr, training=training)
        nirs_out = self.brite(nirs, training=training)
        tobii_out = self.tobii(et, training=training)
        concat = tf.concat([ppg_out, gsr_out, nirs_out, tobii_out], axis=1)
        x = self.D1(concat)
        x = self.BN1(x, training=training)
        x = self.D2(x)
        x = tf.math.maximum(alpha*x, x)
        x = self.reshape(x)
        x = self.C1(x)
        x = self.C2(x)
        x = self.BN2(x, training=training)
        x = self.C3(x)
        x = self.flat(x)
        x = self.D3(x)
        x = self.BN3(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D4(x)
        out = self.out(x)
        return out
