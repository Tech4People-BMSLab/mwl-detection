"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-21-07 12:20:58
@Last Modified by:   Tenzing Dolmans
@Description: Small version of "mlp_base.py",
meaning that all layers have half the units.
This file contains bases for PPG, GSR, ET, and fNIRS,
as well as one head model. The bases are all densely connected.
The head model uses only dense layers.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa
from tensorflow.keras import layers # noqa


class PPGBase(tf.keras.Model):
    "Base for the PPG Modality."
    def __init__(self,
                 activation='relu',
                 units=128):
        super(PPGBase, self).__init__()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="P_D_1")
        self.BN1 = layers.BatchNormalization(name="P_N_1")
        self.D2 = layers.Dense(units=units,
                               activation=activation,
                               name="P_D_2")

    def call(self, inputs, alpha=0.1, training=False):
        x = self.D1(inputs)
        x = self.BN1(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D2(x)
        return x


class GSRBase(tf.keras.Model):
    "Base for the GSR Modality."
    def __init__(self,
                 activation='relu',
                 units=128):
        super(GSRBase, self).__init__()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="G_D_1")
        self.BN1 = layers.BatchNormalization(name="P_N_1")
        self.D2 = layers.Dense(units=units,
                               activation=activation,
                               name="G_D_2")

    def call(self, inputs, alpha=0.1, training=False):
        x = self.D1(inputs)
        x = self.BN1(x, training=training)
        x = tf.math.maximum(alpha * x, x)
        x = self.D2(x)
        return x


class BriteBase(tf.keras.Model):
    "Base for the fNIRS Modality."
    def __init__(self,
                 activation='relu',
                 units=1024):
        super(BriteBase, self).__init__()
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_1")
        self.BN1 = layers.BatchNormalization(name="B_N_1")
        self.D2 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_2")
        self.BN2 = layers.BatchNormalization(name="B_N_2")
        self.D3 = layers.Dense(units=units,
                               activation=activation,
                               name="B_D_3")

    def call(self, inputs, alpha=0.1, training=False):
        x = self.flat(inputs)
        x = self.D1(x)
        x = self.BN1(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D2(x)
        x = self.BN2(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D3(x)
        return x


class TobiiBase(tf.keras.Model):
    "Base for the ET Modality."
    def __init__(self,
                 activation='relu',
                 units=512):
        super(TobiiBase, self).__init__()
        self.flat = layers.Flatten()
        self.D1 = layers.Dense(units=units,
                               activation=activation,
                               name="T_D_1")
        self.BN1 = layers.BatchNormalization(name="T_N_1")
        self.D2 = layers.Dense(units=units,
                               activation=activation,
                               name="T_D_2")
        self.BN2 = layers.BatchNormalization(name="T_N_2")
        self.D3 = layers.Dense(units=units,
                               activation=activation,
                               name="T_D_3")

    def call(self, inputs, alpha=0.1, training=False):
        x = self.flat(inputs)
        x = self.D1(x)
        x = self.BN1(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D2(x)
        x = self.BN2(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D3(x)
        return x


class HeadClass(tf.keras.Model):
    """Head class that learns from all bases.
    First dense layer has the name number of units as all bases
    combined have as outputs."""
    def __init__(self,
                 dropout_rate,
                 units=1024):
        super(HeadClass, self).__init__()
        self.ppg = PPGBase()
        self.gsr = GSRBase()
        self.brite = BriteBase()
        self.tobii = TobiiBase()
        self.drop = layers.Dropout(rate=dropout_rate)
        self.D1 = layers.Dense(units=1792, name="H_D_1")
        self.D2 = layers.Dense(units=units, name="H_D_2")
        self.D3 = layers.Dense(units=units/2, name="H_D_1")
        self.D4 = layers.Dense(units=units/4, name="H_D_2")
        self.BN1 = layers.BatchNormalization(name="H_N_1")
        self.BN2 = layers.BatchNormalization(name="H_N_2")
        self.BN3 = layers.BatchNormalization(name="H_N_3")
        self.out = layers.Dense(units=1, activation='sigmoid')

    def call(self, ppg, gsr, nirs, et, alpha=0.1, training=False):
        ppg_out = self.ppg(ppg, training=training)
        gsr_out = self.gsr(gsr, training=training)
        nirs_out = self.brite(nirs, training=training)
        tobii_out = self.tobii(et, training=training)
        concat = tf.concat([ppg_out, gsr_out, nirs_out, tobii_out], axis=1)
        x = self.D1(concat)
        x = self.BN1(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.D2(x)
        x = self.BN2(x, training=training)
        x = tf.math.maximum(alpha*x, x)
        x = self.drop(x, training=training)
        x = self.D3(x)
        x = self.BN3(x, training=training)
        x = self.D4(x)
        out = self.out(x)
        return out
