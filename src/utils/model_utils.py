"""
Module contains helper functions to train models.
Authors: Fabien Tarrade
"""
import re
import os
import json
from timeit import default_timer as timer
import math
import subprocess
import time
from datetime import datetime
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from google.cloud import storage
import numpy as np
import joblib

def save_model(estimator, gcspath):

    gcspath = re.sub(r'^gs:\/\/', '', gcspath)

    if len(gcspath.split('/')) < 2:
        return 'ERROR: invalid path --> ' + gcspath

    # Instantiates a client
    storage_client = storage.Client()

    # get the bucket
    bucket = storage_client.get_bucket(gcspath.split('/')[0])

    # extract the model
    model = 'model.joblib'
    joblib.dump(estimator, model)

    # save the model
    model_path = os.path.join('/'.join(gcspath.split('/')[1:]),
                              datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'),
                              model)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model)

    return 'gs://' + gcspath.split('/')[0] + model_path


def manual_decay(epoch):
    """
    Manual decay: fixed learning rates for specific epochs
    Functions for decaying the learning rate.
    Manually set decay
    """
    def manual_decay_fn():
        if epoch < 3:
            return 1e-3
        elif 3 <= epoch < 7:
            return 1e-4
        else:
            return 1e-5

    return manual_decay_fn


def exponential_decay(lr0, s):
    """
    Create exponential decay: reduce learning rate by s every specified iteration.

    Parameters
    ----------
    lr0 : float
        initial learning rate
    s: float
        decay rate, e.g. 0.9 (mostly higher than in other methods)

    Returns
    -------
    exponential_decay_fn: float
        exponential decay
    """
    def exponential_decay_fn(steps_per_epoch):
        return lr0 * 0.1**(steps_per_epoch / s)
    return exponential_decay_fn


def step_decay(lr0, s, epochs_drop=1.0):
    """
    Create stepwise decay: Drop learning rate by half (s) every specified iteration.

    Parameters
    ----------
    lr0 : float
        initial learning rate
    s: float
        decay rate, e.g. 0.5, choose lower s than for other decays
    epochs_drop: float
        step size

    Returns
    -------
    step_decay_fn: float
        stepwise decay
    """
    # initial_lrate = 0.1
    # drop = 0.5
    # epochs_drop = 1.0
    def step_decay_fn(steps_per_epoch):
        return lr0 * math.pow(s, math.floor((1 + steps_per_epoch) / epochs_drop))
    return step_decay_fn


def time_decay(lr0, s):
    """
    Create time-based decay: update the learning rate by a decreasing factor each specified iteration.

    Parameters
    ----------
    lr0 : float
        initial learning rate
    s: float
         decay rate, typically between 0.5 and 0.9

    Returns
    -------
    time_decay_fn: float
        time-based decay
    """
    def time_decay_fn(steps_per_epoch):
        return lr0 / (1 + s * steps_per_epoch)
    return time_decay_fn


def no_decay(lr0):
    """
    Create the initial learning rate to ensure that it stays constant.

    Parameters
    ----------
    lr0 : float
        learning rate

    Returns
    -------
    no_decay_fn: float
        learning rate
    """
    def no_decay_fn(steps_per_epoch):
        return lr0
    return no_decay_fn


def test_decay(lr0):
    """
    Create test function to investigate the effect of the decay.

    Parameters
    ----------
    lr0 : float
        learning rate

    Returns
    -------
    test_decay_fn: float
         learning rate
    """
    def test_decay_fn(epoch):
        # if epoch==1:
        #     return lr0
        # else:
        #     return float(0)
        return float(0)
    return test_decay_fn


########################################################################################
# class LearningRateScheduler(tf.keras.callbacks.Callback):
#    '''Learning rate scheduler which sets the learning rate according to schedule.
#    Arguments:
#      schedule: a function that takes an epoch index
#          (integer, indexed from 0) and current learning rate
#          as inputs and returns a new learning rate as output (float).
#    '''
#    def __init__(self, schedule, verbose=1):
#        super(LearningRateScheduler, self).__init__()
#        self.schedule = schedule
#        self.verbose = verbose
#    def on_batch_begin(self, batch, logs=None):
#        #self.verbose = 1
#        if not hasattr(self.model.optimizer, 'lr'):
#            raise ValueError('Optimizer must have a "lr" attribute.')
#        # Get the current learning rate from model's optimizer.
#        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#        # Call schedule function to get the scheduled learning rate.
#        scheduled_lr = self.schedule(lr)
#        #scheduled_lr = self.schedule(epoch, lr)
#        # Set the value back to the optimizer before this epoch starts
#        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
#        print('\nEpoch %05d: Learning rate is %6.4f.' % (batch, scheduled_lr))
#        current_step = self.model.optimizer.iterations.numpy()
#        current_lr = tf.summary.scalar("learning_rate",
#        self.model.optimizer._decayed_lr(var_dtype=tf.float32),
#        step=current_step)
#        print('tf.summary output: ', current_lr)
########################################################################################

# ---------------------------------------------------------------------------------------
# Applying the learning rate scheduler after a specified number of batches and not per epoch

class LearningRateSchedulerPerBatch(tf.keras.callbacks.LearningRateScheduler):
    """
    Callback class to modify the default learning rate scheduler to operate each specified number of batches

    Explanation: The batch counter gets reset each epoch which means that a default counter needs to be
    implemented that counts onwards. If self.k gets updated after each batch, the learning rate after the third batch
    will have already been decreased the third time which is not what we want. Therefore, we added another counter k
    that does not influence the learning rate scheduler. N is defined in model.py and specifies the number of batches
    after which the learning rate gets updated.
    """

    def __init__(self, schedule, N, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__(schedule, verbose)
        self.count = 0  # Global batch index (the regular batch argument refers to the batch index within the epoch)
        self.N = N
        self.k = 1  # another counter that counts the number of batches that have run through
        self.all_lr = []

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        if self.k % self.N == 0:
            print('\nprinting LR in schedule class (before batch_begin): {}, k: {}'.format(
                self.model.optimizer.lr.numpy(),
                self.k))
            super(LearningRateSchedulerPerBatch, self).on_epoch_begin(self.count, logs)
            print('\nprinting LR in schedule class (batch_begin): {}, k: {}'.format(
                self.model.optimizer.lr.numpy(),
                self.k))
            # tf.keras.backend.set_value(self.model.optimizer.lr,
            # float(tf.keras.backend.get_value(self.model.optimizer.lr)))

    def on_batch_end(self, batch, logs=None):
        if self.k % self.N == 0:
            super(LearningRateSchedulerPerBatch, self).on_epoch_end(self.count, logs)
            print('\nprinting LR in schedule class (batch_end): {}, k: {}'.format(
                self.model.optimizer.lr.numpy(),
                self.k))
            # tf.keras.backend.set_value(self.model.optimizer.lr,
            # float(tf.keras.backend.get_value(self.model.optimizer.lr)))

            self.count += 1
        self.k += 1

        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

# -------------------------------------------------------------------------------------

# Callback for printing the LR at the end of each epoch.
# class PrintLR(tf.keras.callbacks.Callback):
#    def on_batch_end(self, batch, logs=None):
#    def on_epoch_end(self, epoch, logs=None):
#    def on_epoch_end(self, epoch, model, logs=None)
#        # self.all_lr.append(self.model.optimizer.lr.numpy())
#        lr_schedule = getattr(self.model.optimizer, "lr", None)
#        print('\nLearning rate for batch {} is {}'.format(batch + 1, self.model.optimizer.lr.numpy()))
#        print(lr_schedule)
# Callback for storing the learning rates each time it changes
# This second class was needed since we didn't find a way to reference tf.keras.callbacks.Callback in the first class


class LR_per_step(tf.keras.callbacks.Callback):
    """
    Callback to provide LR per step
    """

    def on_train_begin(self, logs={}):
        self.all_lr = []
        self.all_lr_alternative = []
        self.all_lr_logs = []
        print('training begin: printing logs: \n', logs)

    def on_epoch_begin(self, epoch, logs={}):
        print('epoch begin: printing logs: \n', logs)

    def on_batch_begin(self, batch, logs={}):
        print('\nbatch begin: printing logs: \n', logs)

        lr_batch = self.model.optimizer.lr.numpy()
        self.all_lr.append(lr_batch)
        print('\nLearning rate for batch {} is {}'.format(batch + 1, lr_batch))

        lr_alternative = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        print('\n=> alternative learning rate for batch {} is {}'.format(batch + 1, lr_alternative))
        self.all_lr_alternative.append(lr_alternative)

        # lr_logs = logs['lr']
        # lr_logs_alt = logs.get('lr')
        # print('\n=> begin: learning rate in the logs for batch {} is {}. alt: {}'.format(
        # batch+1,
        # lr_logs,
        # lr_logs_alt))

    def on_batch_end(self, batch, logs={}):
        print('batch end: printing logs: \n', logs)
        print('eval print: ', tf.keras.backend.eval(self.model.optimizer.lr))

        lr_logs = logs['lr']
        lr_logs_alt = logs.get('lr')
        print('\n=> end: learning rate in the logs for batch {} is {}. alt: {}'.format(batch + 1, lr_logs, lr_logs_alt))
        self.all_lr_logs.append(lr_logs)


class History_per_step(tf.keras.callbacks.Callback):
    """
    Callback to print validation accuracy after N epochs
    """

    def __init__(self, validation_data, N):
        self.validation_data = validation_data
        self.N = N
        self.batch = 1

    def on_train_begin(self, validation_data, logs={}):
        self.steps = []
        self.losses = []
        self.accuracies = []
        self.val_steps = []
        self.val_losses = []
        self.val_accuracies = []
        self.timed_validation = []

    def on_train_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))
        self.steps.append(self.batch)
        print('\n training set -> batch:{} loss:{} and acc: {}'.format(self.batch, logs.get('loss'),
                                                                       logs.get('accuracy')))

        if self.batch % self.N == 0:
            time_start = timer()
            loss_val, acc_val = self.model.evaluate(self.validation_data, verbose=0)
            self.val_losses.append(loss_val)
            self.val_accuracies.append(acc_val)
            self.val_steps.append(self.batch)
            print('\n validation set -> batch:{} val loss:{} and val acc: {}'.format(self.batch, loss_val, acc_val))
            time_valid = timer() - time_start
            self.timed_validation.append(time_valid)
            print('\n validation has lasted for: {} seconds'.format(time_valid))

        self.batch += 1

    def on_test_batch_end(self, batch, logs={}):
        # print('{}\n'.format(logs))
        return


class HP_metric(tf.keras.callbacks.Callback):
    """
    Callback to print validation accuracy after N epochs
    """

    def __init__(self, name_metric):
        self.name_metric = name_metric

    def on_epoch_end(self, epoch, logs={}):
        tf.summary.scalar(self.name_metric, logs.get('accuracy'), step=epoch)
        print('{} : {} epoch {} \n'.format(self.name_metric, logs.get('accuracy'), epoch))
        return


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Class to time each epoch
    """

    def __init__(self, logs={}):
        self.timing_epoch = []
        self.timing_valid = []

    def on_test_begin(self, logs={}):
        self.starttime_valid = timer()

    def on_test_end(self, logs={}):
        self.timing_valid.append(timer() - self.starttime_valid)

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.timing_epoch.append(timer() - self.starttime)


class History_trained_model(object):
    """
    Class to save history from Keras
    """

    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


class History_per_steps_trained_model(object):
    """
    Class to save custom history created using callback
    """

    def __init__(
            self,
            steps,
            losses,
            accuracies,
            val_steps,
            val_losses,
            val_accuracies,
            all_lr,
            all_lr_alternative,
            all_lr_logs
    ):
        self.steps = steps
        self.losses = losses
        self.accuracies = accuracies
        self.val_steps = val_steps
        self.val_losses = val_losses
        self.val_accuracies = val_accuracies
        self.all_learning_rates = all_lr
        self.all_lr_alternative = all_lr_alternative
        self.all_lr_logs = all_lr_logs


def load_data_tensorboard(path):
    """
    Load data collected for TensorBoard.

    Parameters
    ----------
    path : str
        path to TensorBoard files

    Returns
    -------
    data: json
         TensorBoard data in a readable format
    """
    event_acc = event_accumulator.EventAccumulator(path)
    event_acc.Reload()
    data = {}

    for tag in sorted(event_acc.Tags()["scalars"]):
        x, y = [], []
        for scalar_event in event_acc.Scalars(tag):
            x.append(scalar_event.step)
            y.append(scalar_event.value)
        data[tag] = (np.asarray(x), np.asarray(y))
    return data


def copy_local_directory_to_gcs(local_path, bucket_name, gcs_path):
    """
    Recursively copy a directory of files to GCS.

    Parameters
    ----------
    local_path : str
        path to the local path. local_path should be a directory and not have a trailing slash.
    bucket_name: str
        name of the gcs bucket
    gcs_path: str
        path of the gcs
    Returns
    -------
    None
    """
    assert os.path.isdir(local_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_path):
        for name in files:
            local_file = os.path.join(root, name)
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            print(remote_path)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
            print('copy of the file(s) on gs:// done !')


def download_blob(bucket_name, blob_name, destination_file_name):
    """
    Download a blob from the bucket.

    Parameters
    ----------
    bucket_name : str
        bucket's name
    blob_name: str
        blob's name with the following format blob/object-name
    destination_file_name: str
        destimation and name of the file in the following format: local/path/to/file
    Returns
    -------
    None
    """
    # bucket_name = "your-bucket-name"
    # blob_name = "blob/object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blobs = storage_client.list_blobs(bucket, prefix='pretrained_model/bert-base-multilingual-uncased/')
    os.makedirs(destination_file_name + '/' + blob_name, exist_ok=True)
    for blob in blobs:
        print(blob.name)
        blob.download_to_filename(blob.name)

    print('blob {} downloaded to {}'.format(blob_name, destination_file_name + '/' + blob_name))


def get_trial_id():
    """
    Get trial from GCP env variable

    Parameters
    ----------
    None

    Returns
    -------
    name: str
         prepend trial_id_ if trial_id exist
    """
    suffix = json.loads(os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    if suffix == '':
        return suffix

    return 'trial_id_' + suffix


def create_module_tar_archive(model_name):
    # create the package
    process = subprocess.Popen(['python', 'setup.py', 'sdist'], cwd=os.environ['DIR_PROJ'], shell=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait for the process to terminate
    for line in process.stderr:
        print(line.decode('utf8').replace('\n', ''))
    for line in process.stdout:
        print(line.decode('utf8').replace('\n', ''))

    path_package = ''
    name_package = ''
    for root, dirs, files in os.walk(os.environ['DIR_PROJ'] + '/dist/'):
        for filename in files:
            print(root.split('/')[-4] + '/' + filename)
            print('Last modified: {}'.format(time.ctime(os.path.getmtime(root + '/' + filename))))
            print('Created: {}'.format(time.ctime(os.path.getctime(root + '/' + filename))))
            name_package = filename
            path_package = root + '/' + name_package
    bucket_name = os.environ['BUCKET_STAGING_NAME']
    output_folder = model_name + '_' + datetime.now().strftime("%Y_%m_%d_%H%M%S")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_folder + '/' + name_package)
    blob.upload_from_filename(path_package)

    path_package_gcs = 'gs://' + os.environ['BUCKET_STAGING_NAME'] + '/' + output_folder + '/' + name_package

    return path_package_gcs
