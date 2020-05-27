import tensorflow as tf
from transformers import (
    __version__,
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
import sys
from datetime import timedelta
import hypertune
from tensorboard.plugins.hparams import api as hp
import math


def build_dataset(input_tfrecords, batch_size, shuffle_buffer=2048):

    #print('debug1:', input_tfrecords)
    #file_pattern = tf.io.gfile.glob(input_tfrecords+'/*.tfrecord')
    #pattern = input_tfrecords+'/*.tfrecord'
    file_pattern = input_tfrecords+'/*.tfrecord'
    #print('debug 1:', file_pattern)
    print('test 2-1:', list(tf.data.Dataset.list_files(tf.io.gfile.glob(file_pattern))))
    #print('debug 2:', list(tf.data.Dataset.list_files(file_pattern)))
    # standard
    #dataset = tf.data.TFRecordDataset(file_pattern)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.batch(batch_size)
    #return dataset

    # standard 1
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print('debug 3:')
    for i in dataset:
        print('i=',i)
        break
    print('debug 4:')
    return dataset

    # standard 2
    #dataset = tf.data.TFRecordDataset(file_pattern)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #return dataset

    # best way ?
    #dataset = tf.data.Dataset.list_files(file_pattern,
    #                                     shuffle=True,
    #                                     seed=None
    #                                     )
    #dataset = dataset.interleave(tf.data.TFRecordDataset,
    #                             cycle_length=tf.data.experimental.AUTOTUNE,
    #                             num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #                             deterministic=False)
    #dataset = dataset.shuffle(shuffle_buffer)
    #dataset = dataset.map(pp.parse_tfrecord_glue_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.batch(batch_size)
    #dataset = dataset.cache()
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #return dataset

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


def train_and_evaluate(model, num_epochs, steps_per_epoch, train_data, validation_steps, eval_data, output_dir, n_steps_history, FLAGS, decay_type, learning_rate=3e-5, s=1, n_batch_decay=1):
    """Compiles keras model and loads data into it for training."""
    logging.info('training the model ...')
    model_callbacks = []

    # create meta data dictionary
    dict_model = {}
    dict_data = {}
    dict_parameter = {}
    dict_hardware = {}
    dict_results = {}
    dict_type_job = {}
    dict_software = {}

    if FLAGS.is_hyperparameter_tuning:
        # get trial ID
        suffix = mu.get_trial_id()

        if suffix == '':
            logging.error('No trial ID for hyper parameter job!')
            FLAGS.is_hyperparameter_tuning=False

    if output_dir:
        # tensorflow callback
        log_dir = os.path.join(output_dir, 'tensorboard')
        if FLAGS.is_hyperparameter_tuning:
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
        if not FLAGS.is_hyperparameter_tuning:
            # not saving model during hyper parameter tuning
            #checkpoint_dir = os.path.join(checkpoint_dir, suffix)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch:02d}')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                                     verbose=1,
                                                                     save_weights_only=True)
            model_callbacks.append(checkpoint_callback)

        # decay learning rate callback
        #---------------------------------------------------------------------------------------------------------------
        # exponential decay function
        #def exponential_decay(lr0, s):
        #    def exponential_decay_fn(steps_per_epoch):
        #        return lr0 * 0.1**(steps_per_epoch / s)
        #    return exponential_decay_fn
        
        # code snippet to make the switching between different learning rate decays possible
        if decay_type=='exponential':
            decay_fn = mu.exponential_decay(lr0=learning_rate, s=s)
        elif decay_type=='stepwise':
            decay_fn = mu.step_decay(lr0=learning_rate, s=s)
        elif decay_type=='timebased':
            decay_fn = mu.time_decay(lr0=learning_rate, s=s)
        else:
            decay_fn = mu.no_decay(lr0=learning_rate)
            
        #exponential_decay_fn = mu.exponential_decay(lr0=learning_rate, s=s)
        #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn, verbose=1)
        #model_callbacks.append(lr_scheduler)
        
        # added these two lines for batch updates
        lr_decay_batch = mu.LearningRateSchedulerPerBatch(decay_fn, n_batch_decay, verbose=1)
        #lr_decay_batch = mu.LearningRateSchedulerPerBatch(exponential_decay_fn, n_batch_decay, verbose=0)
                    #lambda step: ((learning_rate - min_learning_rate) * decay_rate ** step + min_learning_rate))
        model_callbacks.append(lr_decay_batch)
        
        #print_lr = mu.PrintLR()
        #model_callbacks.append(mu.PrintLR())
        #---------------------------------------------------------------------------------------------------------------
  
    # callback to store all the learning rates
    #all_learning_rates = mu.LearningRateSchedulerPerBatch(model.optimizer, n_steps_history)
    all_learning_rates = mu.LR_per_step()
##    all_learning_rates = mu.LR_per_step(model.optimizer)
    model_callbacks.append(all_learning_rates)
    
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
    logging.info('timing per validation:\n{}'.format(list(map(lambda x: str(timedelta(seconds=round(x))),timing.timing_valid))))
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

    if FLAGS.is_hyperparameter_tuning:
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

        hparams_dir = os.path.join(output_dir, 'hparams_tuning')
        with tf.summary.create_file_writer(hparams_dir).as_default():
            hp.hparams_config(
                hparams=list_hp,
                metrics=[hp.Metric(metric_accuracy, display_name='Accuracy')],
            )
        if FLAGS.is_hyperparameter_tuning:
            hparams_dir = os.path.join(hparams_dir, suffix)

        with tf.summary.create_file_writer(hparams_dir).as_default():
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
                                                                    histories_per_step.val_accuracies,
                                                                    all_learning_rates.all_lr,
                                                                    all_learning_rates.all_lr_alternative)
        pickle.dump(model_history_per_step, file, pickle.HIGHEST_PROTOCOL)

    if output_dir:
        # save the model
        savemodel_path = os.path.join(output_dir, 'saved_model')
        if not FLAGS.is_hyperparameter_tuning:
            # not saving model during hyper parameter tuning
            #savemodel_path = os.path.join(savemodel_path, suffix)
            model.save(os.path.join(savemodel_path, model.name))

        # save history
        search = re.search('gs://(.*?)/(.*)', output_dir)
        if search is not None:
            bucket_name = search.group(1)
            blob_name = search.group(2)
            output_folder=blob_name+'/history'
            if FLAGS.is_hyperparameter_tuning:
                output_folder = os.path.join(output_folder, suffix)
            mu.copy_local_directory_to_gcs(history_dir, bucket_name, output_folder)

    # add meta data
    dict_model['pretrained_transformer_model'] = FLAGS.pretrained_model_dir
    dict_model['num_classes'] = FLAGS.num_classes

    dict_data['train'] = FLAGS.input_train_tfrecords
    dict_data['eval'] = FLAGS.input_eval_tfrecords

    dict_parameter['use_decay_learning_rate'] = FLAGS.use_decay_learning_rate
    dict_parameter['epochs'] = FLAGS.epochs
    dict_parameter['steps_per_epoch_train'] = FLAGS.steps_per_epoch_train
    dict_parameter['steps_per_epoch_eval'] = FLAGS.steps_per_epoch_eval
    dict_parameter['n_steps_history'] = FLAGS.n_steps_history
    dict_parameter['batch_size_train'] = FLAGS.batch_size_train
    dict_parameter['batch_size_eval'] = FLAGS.batch_size_eval
    dict_parameter['learning_rate'] = FLAGS.learning_rate
    dict_parameter['epsilon'] = FLAGS.epsilon

    dict_hardware['is_tpu'] = FLAGS.use_tpu

    dict_results['tensorflow'] = 1
    dict_results['tensorflow'] = 1
    dict_results['tensorflow'] = 1

    dict_type_job['is_hyperparameter_tuning'] = FLAGS.is_hyperparameter_tuning
    dict_type_job['is_tpu'] = FLAGS.use_tpu

    dict_software['tensorflow'] = tf.__version__
    dict_software['transformer'] = __version__
    dict_software['python'] = sys.version

    # aggregate dictionaries
    dict_all = {'model': dict_model,
                'data': dict_data,
                'parameter': dict_parameter,
                'hardware': dict_hardware,
                'results': dict_results,
                'type_job': dict_type_job,
                'software': dict_software}

    # save metadata
    search = re.search('gs://(.*?)/(.*)', output_dir)
    if search is not None:
        bucket_name = search.group(1)
        blob_name = search.group(2)
        output_folder = blob_name + '/metadata'

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(output_folder+'/model_job_metadata.json')
        blob.upload_from_string(
            data=json.dumps(dict_all),
            content_type='application/json'
        )