"""
-*- coding: utf-8 -*-
@Author: Tenzing Dolmans
@Date:   2020-07-17 10:52:26
@Last Modified by:   Tenzing Dolmans
@Last Modified time: 2020-09-08 14:41:34
@Description: Trainer class that is used to train models.
Also contains evaluation functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tfrecord_reader import tfrecord_dataset
from Models.mlp_base import HeadClass as MlpClass  # noqa
from Models.small_mlp_base import HeadClass as sMlpClass  # noqa
from Models.literature_base import HeadClass as LitClass  # noqa
from Models.small_literature_base import HeadClass as sLitClass  # noqa
import tensorflow as tf
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Trainer():
    """Initialise a Trainer class for classifying MWL. Methods included
    are used to load TFRecord datasets, cycle through LR, and train."""
    def __init__(self,
                 num_epochs=3,
                 batch_size=8,
                 dropout_rate=0.2,
                 min_lr=0.0007,
                 max_lr=0.1,
                 min_mom=0.85,
                 max_mom=0.95):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # Note that a standard model is defined here
        self.model = sMlpClass(dropout_rate)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom

    def get_dataset(self):
        """Loads a TFRecord dataset from a file, performs train & test
        splitting and shuffling. See Tensorflow Docs for more methods.
        See "hpo_search.py" before deciding whether to return one dataset
        or a split train/test set."""
        file = 'ZebraDatasetV1.03.tfrecord'
        raw_dataset = tf.data.TFRecordDataset(file)
        dataset = tfrecord_dataset(self.batch_size, raw_dataset)
        dataset = dataset.shuffle(buffer_size=4200,
                                  reshuffle_each_iteration=False)
        # Split Train (90%), Test (10%)
        train_dataset = dataset.take(469)
        test_dataset = dataset.skip(470)
        train_dataset = train_dataset.shuffle(buffer_size=4200,
                                              reshuffle_each_iteration=True)
        test_dataset = test_dataset.shuffle(buffer_size=4200,
                                            reshuffle_each_iteration=True)
        return train_dataset, test_dataset

    def get_step_size(self, train_dataset):
        """Returns step size based on the dataset and num_epochs,
        which is used to calculate the learning rate."""
        self.num_iterations = 0
        for batch in train_dataset:
            self.num_iterations += 1
        step_size = (self.num_epochs * self.num_iterations / 2) * 0.85
        return step_size

    def get_learning_rate(self, iterations_seen):
        """Returns learning rate and momentum based on progession thus far.
        https://github.com/nachiket273/One_Cycle_Policy/blob/master/OneCycle.py"""  # noqa
        cycle = np.floor(1 + iterations_seen / (2 * self.step_size))
        x = np.abs(iterations_seen / self.step_size - 2 * cycle + 1)
        x = np.maximum(0, 1 - x)
        div = 100
        if iterations_seen > 2 * self.step_size:
            ratio = (iterations_seen - 2 * self.step_size) / (
                self.num_iterations * self.num_epochs - 2 * self.step_size)
            _lr = self.min_lr * (1 - ratio * (1 - 1 / div))
            _mom = self.max_mom
        else:
            _lr = self.min_lr + x * (self.max_lr - self.min_lr)
            _mom = self.max_mom - x * (self.max_mom - self.min_mom)
        return _lr, _mom

    def forward_pass(self, gsr, ppg, nirs, et, training):
        """Simple forward pass of all data through the model.
        Is used in both the training and validation/testing stages."""
        y_pred = self.model(gsr=gsr, ppg=ppg, nirs=nirs, et=et,
                            training=training)
        return y_pred

    @tf.function
    def train_step(self, gsr, ppg, nirs, et, label):
        """Single training step, is done for each batch in every epoch."""
        with tf.GradientTape() as tape:
            y_pred = self.forward_pass(
                gsr=gsr, ppg=ppg, nirs=nirs, et=et, training=True)
            loss_value = self.loss_object(y_true=label, y_pred=y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return y_pred

    def training_loop(self, train_dataset, test_dataset):
        """Loop that wraps train_step() in epochs and batches.
        Resets iterations_seen on call, returns many metrics.
        NOTE: Accuracy is simply the absolute difference between
        the predicted vs true label. Hence, a lower accuracy is better."""
        self.iterations_seen = 0
        self.step_size = self.get_step_size(train_dataset)

        train_loss_results = []
        train_accuracy_results = []
        test_loss_results = []
        test_accuracy_results = []
        test_labels = []
        test_predictions = []

        for epoch in range(self.num_epochs):
            epoch_loss_avg = tf.keras.metrics.MeanSquaredError()
            epoch_accuracy = []
            test_loss_avg = tf.keras.metrics.MeanSquaredError()
            test_accuracy = []
            epoch_labels = []
            epoch_predicts = []

            # --- Training --- Optimise the model
            for ppg, gsr, nirs, et, label, partno in train_dataset:
                y_pred = self.train_step(
                    gsr=gsr, ppg=ppg, nirs=nirs, et=et, label=label)
                _lr, _mom = self.get_learning_rate(self.iterations_seen)
                self.optimizer.learning_rate = _lr
                self.optimizer.momentum = _mom
                self.iterations_seen += 1
                epoch_loss_avg.update_state(label, y_pred)
                epoch_accuracy.extend(abs(y_pred - label))

            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(np.mean(epoch_accuracy))

            # --- Testing --- Only a forward pass, does not update gradients
            for ppg, gsr, nirs, et, label, partno in test_dataset:
                y_pred = self.forward_pass(
                    gsr=gsr, ppg=ppg, nirs=nirs, et=et, training=False)
                test_loss_avg.update_state(label, y_pred)
                test_accuracy.extend(abs(y_pred - label))
                epoch_labels.extend(label)
                epoch_predicts.extend(y_pred)

            # Keep track of all new metrics
            test_labels.append(epoch_labels)
            test_predictions.append(epoch_predicts)
            test_loss_results.append(test_loss_avg.result())
            test_accuracy_results.append(np.mean(test_accuracy))

            print('Epoch {:03d}: Loss {:.4f}, Accuracy {:.4f}',
                  'Test Loss: {:.4f}, Test Accuracy: {:.4f}'
                  .format(
                      epoch + 1, epoch_loss_avg.result(),
                      np.mean(epoch_accuracy), test_loss_avg.result(),
                      np.mean(test_accuracy)))

        # Objective is what HPO will seek to optimise. Adjust as needed.
        objective = test_accuracy_results[-1]
        return (train_loss_results,
                train_accuracy_results,
                test_loss_results,
                test_accuracy_results,
                test_labels,
                test_predictions, objective)


"""Evaluation Functions"""


def plot_loss(train_loss_results,
              train_accuracy_results,
              test_loss_results,
              test_accuracy_results):
    """Plots the progression of traing and testing loss + accuracy."""
    fig, axes = plt.subplots(4, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Train Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Train Accuracy", fontsize=14)
    axes[1].plot(train_accuracy_results)

    axes[2].set_ylabel("Test Loss", fontsize=14)
    axes[2].plot(test_loss_results)

    axes[3].set_xlabel("Epoch", fontsize=14)
    axes[3].set_ylabel("Test Accuracy", fontsize=14)
    axes[3].plot(test_accuracy_results)
    plt.show()


def evaluate_performance(test_labels, test_predictions):
    """Select one epoch in test_labels and test_predictions to work with.
    Then creates a dataframe that contains all labels, predictions,
    and differences. DFs for (un)acceptable results are also made and returned.
    These are later used for plotting."""
    label_results = [l[0].numpy() for l in test_labels[-1]]
    predicted_results = [l[0].numpy() for l in test_predictions[-1]]
    d = {'labels': label_results, 'predicted': predicted_results}
    df = pd.DataFrame(data=d)
    df['difference'] = np.array(predicted_results) - np.array(label_results)
    df = df.sort_values(by='labels')

    # Prevalance refers to how common each label is
    prevalence = df.groupby('labels')['predicted'].nunique()

    # Acceptability bound is currently one label (1/6)
    acceptable = df[abs(df.predicted - df.labels) <= 1/6]
    unacceptable = df[abs(df.predicted - df.labels) > 1/6]

    # Accuracy is the proportion of predictions that are within 1/6
    acc = len(acceptable)/len(df.difference)
    print('Mean difference between label and prediction: {}'
          .format(np.mean(df.difference)))
    print('Percentage of predictions that are within 1.5 levels: ', acc*100)
    return df, prevalence, df.difference, acceptable, unacceptable


def level_accuracies(df, prevalence):
    """returns workable lists of results that are looped over
    while plotting histograms."""
    _all = [df[df.labels == prevalence.keys()[i]]
            for i in range(len(prevalence))]
    _accept = [acceptable[acceptable.labels == prevalence.keys()[i]]
               for i in range(len(prevalence))]
    _unaccept = [unacceptable[unacceptable.labels == prevalence.keys()[i]]
                 for i in range(len(prevalence))]
    return _all, _accept, _unaccept


def plot_all_histograms(data, prevalence):
    """Plots as many histograms as there are lists in input data. The histograms
    are separated by label and contain a vertical line of the label and bins of
    predictions around said label. """
    fig, ax = plt.subplots(len(data), figsize=(20, 20))
    num_bins = 15

    for i, current in enumerate(data):
        mu = np.mean(current.predicted)
        sigma = np.std(current.predicted) + 1e-4
        x = current.predicted.values
        print('Level: {} Mu: {:.3f} Sigma: {:.3f}'.format(i + 1, mu, sigma))
        ax[i].set_xlim(left=-0.1, right=1.1)
        ax[i].set_xlabel('Predicted Label')
        ax[i].set_ylabel('Probability density')
        ax[i].set_title(r'Histogram of predictions: $\mu={:.3f}$',
                        '$\\sigma={:.3f}$'.format(mu, sigma))

        # the histogram of the data
        n, bins, patches = ax[i].hist(x, num_bins, density=True)

        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * (1 / sigma * (bins - mu))**2))
        ax[i].plot(bins, y, '--')
        ax[i].axvline(prevalence.keys()[i])

    fig.tight_layout()  # Tweak spacing to prevent clipping of ylabel
    plt.show()


def plot_histogram(difference, acceptable, unacceptable):
    """Plots a single histogram that contains all predictions'
    relative offset. The slimmer and closer to zero the histogram,
    the better the results."""
    fig, ax = plt.subplots()
    num_bins = 30

    """All differences"""
    mu0 = np.mean(difference.predicted - difference.labels)
    sigma0 = np.std(difference.predicted - difference.labels) + 1e-4
    x0 = df.difference

    ax.set_xlim(left=-1, right=1)
    ax.set_xlabel('Predicted Label Difference')
    ax.set_ylabel('Number of Predictions')
    ax.set_title(r'Histogram of Predictions: $\mu={:.3f}$, $\sigma={:.3f}$'
                 .format(mu0, sigma0))

    """Differences within an acceptable range"""
    mu1 = np.mean(acceptable.predicted - acceptable.labels)
    sigma1 = np.std(acceptable.predicted - acceptable.labels) + 1e-4
    x1 = acceptable.predicted - acceptable.labels

    """Differences outside an acceptable range"""
    mu2 = np.mean(unacceptable.predicted - unacceptable.labels)
    sigma2 = np.std(unacceptable.predicted - unacceptable.labels) + 1e-4
    x2 = unacceptable.predicted - unacceptable.labels

    """Plot the Histogram(s)"""
    n0, bins0, patches0 = ax.hist(x0, num_bins, density=False, color='blue')
    n1, bins1, patches1 = ax.hist(x1, num_bins, density=False, color='green')
    n2, bins2, patches2 = ax.hist(x2, num_bins, density=False, color='red')

    """Add 'best fit' line(s)"""
    y0 = ((1 / (np.sqrt(2 * np.pi) * sigma0)) * np.exp(
        -0.5 * (1 / sigma0 * (bins0 - mu0))**2))
    ax.plot(bins0, y0, '-')

    y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(
        -0.5 * (1 / sigma1 * (bins1 - mu1))**2))
    ax.plot(bins1, y1, '--')

    y2 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(
        -0.5 * (1 / sigma2 * (bins2 - mu2))**2))
    ax.plot(bins2, y2, '--')

    """Vertical lines to indicate acceptable bounds."""
    ax.axvline(-1/6)
    ax.axvline(1/6)
    fig.tight_layout()  # Tweak spacing to prevent clipping of ylabel
    plt.show()


if __name__ == "__main__":
    # Select a GPU to train with
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """TRAINING"""
    # Give the HPs that were found by HPO
    trainer = Trainer(num_epochs=25,
                      batch_size=8,
                      dropout_rate=0.1789,
                      min_lr=0.000000796,
                      max_lr=0.0000341,
                      min_mom=0.7403,
                      max_mom=0.7985)
    train_dataset, test_dataset = trainer.get_dataset()
    (tr_loss, tr_acc, te_loss, te_acc,
     te_labels, te_preds, objective) = trainer.training_loop(
        train_dataset, test_dataset)

    """EVALUATION"""
    (df, prevalence, difference,
     acceptable, unacceptable) = evaluate_performance(te_labels, te_preds)
    _all, _accept, _unaccept = level_accuracies(df, prevalence)
    plot_all_histograms(_all, prevalence)
    plot_histogram(difference, acceptable, unacceptable)
