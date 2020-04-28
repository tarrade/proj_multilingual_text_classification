import os
import datetime
#import joblib
from sklearn.externals import joblib
import numpy as np
import re
import os
from google.cloud import storage
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from google.cloud import storage

def save_model(estimator, gcspath, name):
    
    gcspath = re.sub('^gs:\/\/', '', gcspath)
    
    if len(gcspath.split('/'))<2:
        return 'ERROR: invalid path --> '+gcspath
    
    # Instantiates a client
    storage_client = storage.Client()

    # get the bucket
    bucket = storage_client.get_bucket(gcspath.split('/')[0])
    
    # extract the model
    model = 'model.joblib'
    joblib.dump(estimator, model)
    
    # save the model
    model_path = os.path.join('/'.join(gcspath.split('/')[1:]), datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model)

    return 'gs://'+gcspath.split('/')[0]+model_path

# Function for decaying the learning rate.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, model, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

# Callback to print validation accuracy after N epochs
class History_per_step(tf.keras.callbacks.Callback):

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

    def on_train_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))
        self.steps.append(self.batch)
        print('\n training set -> batch:{} loss:{} and acc: {}'.format(self.batch, logs.get('loss'),
                                                                       logs.get('accuracy')))

        if self.batch % self.N == 0:
            loss_val, acc_val = self.model.evaluate(self.validation_data, verbose=0)
            self.val_losses.append(loss_val)
            self.val_accuracies.append(acc_val)
            self.val_steps.append(self.batch)
            print('\n validation set -> batch:{} val loss:{} and val acc: {}'.format(self.batch, loss_val, acc_val))

        self.batch += 1

    def on_test_batch_end(self, batch, logs={}):
        # print('{}\n'.format(logs))
        return

    def on_epoch_end(self, batch, logs={}):
        # print('{}\n'.format(logs))
        return

# Class to save history from Keras
class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

# Class to save custome history created using callback
class History_per_steps_trained_model(object):
    def __init__(self, steps, losses, accuracies, val_steps, val_losses, val_accuracies):
        self.steps = steps
        self.losses = losses
        self.accuracies = accuracies
        self.val_steps = val_steps
        self.val_losses = val_losses
        self.val_accuracies = val_accuracies

def load_data_tensorboard(path):
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

def copy_local_directory_to_gcs(local_path, bucket, gcs_path):
    """Recursively copy a directory of files to GCS.

    local_path should be a directory and not have a trailing slash.
    """
    assert os.path.isdir(local_path)
    for root, dirs, files in os.walk(local_path):
        for name in files:
            local_file = os.path.join(root, name)
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
            print(remote_path)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
            print('copy of the file on gs:// done !')

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    for file in storage_client.list_blobs(bucket):
        blob = bucket.blob([REMOTE PATH] / filename)
        blob.upload_from_filename(filename)

    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name)
    )
