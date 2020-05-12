import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification,
)
import utils.model_utils as mu
import preprocessing.preprocessing as pp
import os
import glob
import re
import pickle
from absl import logging
import time
import json
from datetime import timedelta
import hypertune
from tensorboard.plugins.hparams import api as hp

def build_dataset(input_tfrecords, batch_size, shuffle_buffer=2048):

    #print('debug1:', input_tfrecords)
    #file_pattern = tf.io.gfile.glob(input_tfrecords+'/*.tfrecord')
    #print('debug2:',  file_pattern)
    #pattern = input_tfrecords+'/*.tfrecord'
    file_pattern = input_tfrecords+'/*.tfrecord'
    #print('test 2-1:', list(tf.data.Dataset.list_files(tf.io.gfile.glob(file_pattern))))
    #print('test 2-2:', list(tf.data.Dataset.list_files(file_pattern)))
    # standard
    #dataset = tf.data.TFRecordDataset(file_pattern)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.batch(batch_size)
    #return dataset

    # standard 1
    #dataset = tf.data.TFRecordDataset(file_pattern)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #return dataset

    # standard 2
    #dataset = tf.data.TFRecordDataset(file_pattern)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #return dataset

    # best way ?
    dataset = tf.data.Dataset.list_files(file_pattern,
                                         shuffle=True,
                                         seed=None
                                         )
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=tf.data.experimental.AUTOTUNE,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 deterministic=False)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

    # standard 4 -> issue: flat  accuracy and loss
    #dataset = tf.data.Dataset.from_tensor_slices(file_pattern)
    #dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                             cycle_length=tf.data.experimental.AUTOTUNE,
    #                             num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #                             deterministic=False)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #return dataset



    # use take(55) to take 55 events or batches
    #return tf.data.Dataset.list_files(
    #    file_pattern
    #).interleave(
    #    tf.data.TFRecordDataset,
    #    cycle_length=tf.data.experimental.AUTOTUNE,
    #    num_parallel_calls=tf.data.experimental.AUTOTUNE
    #).shuffle(
    #    shuffle_buffer
    #).map(
    #    map_func=pp.parse_tfrecord_glue_files,
    #    num_parallel_calls=tf.data.experimental.AUTOTUNE
    #).batch(
    #    batch_size=batch_size,
    #    drop_remainder=True
    #).cache(
    #).prefetch(
    #    tf.data.experimental.AUTOTUNE
    #)

def create_model(pretrained_weights, pretrained_model_dir, num_labels, learning_rate, epsilon):
    """Creates Keras Model for BERT Classification.
    Args:
      pretrained_weights
      pretrained_model_dir
      num_labels
      learning_rate,
      epsilon
    Returns:
      The compiled Keras model
    """

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # metric
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)

    logging.debug('pretrained model\'s files: \n {}'.format(glob.glob(pretrained_model_dir+"/*")))

    # create and compile the Keras model in the context of strategy.scope
    model= TFBertForSequenceClassification.from_pretrained(pretrained_weights,
                                                           num_labels=num_labels,
                                                           cache_dir=pretrained_model_dir)
    #model.layers[-1].activation = tf.keras.activations.softmax
    model._name = 'tf_bert_classification'

    # compile Keras model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])
    return model


def train_and_evaluate(model, num_epochs, steps_per_epoch, train_data, validation_steps, eval_data, output_dir, n_steps_history, FLAGS):
    """Compiles keras model and loads data into it for training."""
    logging.info('training the model ...')
    model_callbacks = []
    suffix = mu.get_trial_id()

    if output_dir:
        # tensorflow callback
        log_dir = os.path.join(output_dir, 'tensorboard')
        if suffix != '':
            log_dir = os.path.join(log_dir, suffix)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              embeddings_freq=1,
                                                              write_graph=True,
                                                              update_freq='batch',
                                                              profile_batch='10, 20')
        model_callbacks.append(tensorboard_callback)

        # checkpoints callback
        checkpoint_dir = os.path.join(output_dir, 'checkpoint_model')
        if suffix != '':
            checkpoint_dir = os.path.join(checkpoint_dir, suffix)
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch:02d}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                                 verbose=1,
                                                                 save_weights_only=True)
        model_callbacks.append(checkpoint_callback)

        # decay learning rate callback
        #decay_callback = tf.keras.callbacks.LearningRateScheduler(mu.decay)
        #model_callbacks.append(decay_callback)

    # callback to create  history per step (not per epoch)
    histories_per_step = mu.History_per_step(eval_data, n_steps_history)
    model_callbacks.append(histories_per_step)

    # callback to time each epoch
    timing = mu.TimingCallback()
    model_callbacks.append(timing)

    # train the model

    # time the function
    start_time = time.time()

    history = model.fit(train_data,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=eval_data,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=model_callbacks)

    # print execution time
    elapsed_time_secs = time.time() - start_time
    logging.info('\nexecution time: {}'.format(timedelta(seconds=round(elapsed_time_secs))))

    logging.info('timing per epoch:\n{}'.format(list(map(lambda x: str(timedelta(seconds=round(x))),timing.timing_epoch))))
    logging.info('sum timing over all epochs:\n{}'.format(timedelta(seconds=round(sum(timing.timing_epoch)))))

    # this is for hyperparameter tuning
    #logging.info('[1] list all files: \n')
    #for root, dirs, files in os.walk("/var/hypertune/"):
    #    # print(root, dirs)
    #    for f in files:
    #        #if 'output.metric' in f:
    #        print(root + f)

    #logging.info('[2] list all files: \n')
    #for root, dirs, files in os.walk("/tmp/hypertune/"):
    #    # print(root, dirs)
    #    for f in files:
    #        #if 'output.metric' in f:
    #        print(root + f)

    #logging.info('hyperparameter tuning "accuracy_train": {}'.format(histories_per_step.accuracies))
    metric_accuracy='accuracy_train'
    value_accuracy=histories_per_step.accuracies[-1]
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=metric_accuracy,
        metric_value=value_accuracy,
        global_step=0)

    #logging.info('[2] list all files: \n')
    #for root, dirs, files in os.walk("/var/hypertune/"):
    #    # print(root, dirs)
    #    for f in files:
    #        #if 'output.metric' in f:
    #        print(root + f)

    #path_metric='/var/hypertune/output.metric'
    #with open(path_metric, 'r') as f:
    #    print(f.read())

    # for hp parameter tuning in TensorBoard
    print(eval('FLAGS.learning_rate'))
    if suffix != '':
        params = json.loads(os.environ.get("TF_CONFIG", "{}")).get("job", {}).get("hyperparameters", {}).get("params", {})
        list_hp = []
        hparams = {}
        for el in params:
            hp_dict = dict(el)
            if hp_dict.get('type') == 'DOUBLE':
                key_hp=hp.HParam(hp_dict.get('parameter_name'),
                                         hp.RealInterval(hp_dict.get('min_value'), hp_dict.get('max_value')))
                list_hp.append(key_hp)
                try:
                    hparams[key_hp] = FLAGS[hp_dict.get('parameter_name')].value
                except KeyError:
                    logging.error('hyperparameter key {} doesn\'t exist'.format(hp_dict.get('parameter_name')))
                # to be deleted
                #hparams[key_hp]=eval('FLAGS.'+hp_dict.get('parameter_name'))

        print(list_hp)
        with tf.summary.create_file_writer(os.path.join(output_dir, 'tensorboard')).as_default():
            hp.hparams_config(
                hparams=list_hp,
                metrics=[hp.Metric(metric_accuracy, display_name='Accuracy')],
            )

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar(metric_accuracy, value_accuracy, step=1)

    # save the history in a file
    search = re.search('gs://(.*?)/(.*)', output_dir)
    if search is not None:
        # temp folder locally and to be  ove on gcp later
        history_dir = os.path.join('./', model.name)
        os.makedirs(history_dir, exist_ok=True)
    else:
        # locally
        history_dir = os.path.join(output_dir, model.name)
        os.makedirs(history_dir, exist_ok=True)
    logging.debug('history_dir: \n {}'.format(history_dir))

    with open(history_dir + '/history', 'wb') as file:
        model_history = mu.History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)

    with open(history_dir + '/history_per_step', 'wb') as file:
        model_history_per_step = mu.History_per_steps_trained_model(histories_per_step.steps,
                                                                    histories_per_step.losses,
                                                                    histories_per_step.accuracies,
                                                                    histories_per_step.val_steps,
                                                                    histories_per_step.val_losses,
                                                                    histories_per_step.val_accuracies)
        pickle.dump(model_history_per_step, file, pickle.HIGHEST_PROTOCOL)

    if output_dir:
        # save the model
        savemodel_path = os.path.join(output_dir, 'saved_model')
        if suffix != '':
            savemodel_path = os.path.join(savemodel_path, suffix)
        model.save(os.path.join(savemodel_path, model.name))

        # save history
        search = re.search('gs://(.*?)/(.*)', output_dir)
        if search is not None:
            bucket_name = search.group(1)
            blob_name = search.group(2)
            output_folder=blob_name+'/history'
            if suffix != '':
                output_folder = os.path.join(output_folder, suffix)
            mu.copy_local_directory_to_gcs(history_dir, bucket_name, output_folder)