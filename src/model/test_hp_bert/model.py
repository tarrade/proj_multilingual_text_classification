import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification,
)
import utils.model_utils as mu
import preprocessing.preprocessing as pp
import glob
from absl import logging
import time
from datetime import timedelta


def build_dataset(input_tfrecords, batch_size, shuffle_buffer=2048):
    file_pattern = input_tfrecords + '/*.tfrecord'
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


def create_model(pretrained_weights, pretrained_model_dir, num_labels, learning_rate, epsilon):
    """
    Creates Keras Model for BERT Classification.
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

    logging.debug('pretrained model\'s files: \n {}'.format(glob.glob(pretrained_model_dir + "/*")))

    # create and compile the Keras model in the context of strategy.scope
    model = TFBertForSequenceClassification.from_pretrained(pretrained_weights,
                                                            num_labels=num_labels,
                                                            cache_dir=pretrained_model_dir)
    # model.layers[-1].activation = tf.keras.activations.softmax
    model._name = 'tf_bert_classification'

    # compile Keras model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])
    return model


def train_and_evaluate(
    model,
    num_epochs,
    steps_per_epoch,
    train_data,
    validation_steps,
    eval_data,
    output_dir,
    n_steps_history,
    FLAGS,
    decay_type,
    learning_rate=3e-5,
    s=1,
    n_batch_decay=1,
    metric_accuracy='metric',
):
    """
    Compiles keras model and loads data into it for training.
    """
    logging.info('training the model ...')
    model_callbacks = []
    activate_tf_summary_hp = True  # False

    if FLAGS.is_hyperparameter_tuning:
        # get trial ID
        suffix = mu.get_trial_id()

        if suffix == '':
            logging.error('No trial ID for hyper parameter job!')
            FLAGS.is_hyperparameter_tuning = False
        else:
            # callback for hp
            logging.info('Creating a callback to store the metric!')
            if activate_tf_summary_hp:
                logging.info('Hp parameters\'s name {}'.format(metric_accuracy))
                hp_metric = mu.HP_metric(metric_accuracy)
                model_callbacks.append(hp_metric)

    # checking model callbacks for
    logging.info('model\'s callback:\n {}'.format(str(model_callbacks)))

    # train the model
    # time the function
    start_time = time.time()

    logging.info('starting model.fit')
    # verbose = 0 (silent)
    # verbose = 1 (progress bar)
    # verbose = 2 (one line per epoch)
    verbose = 1
    model.fit(train_data,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=eval_data,
              validation_steps=validation_steps,
              verbose=verbose,
              callbacks=model_callbacks)

    # print execution time
    elapsed_time_secs = time.time() - start_time
    logging.info('\nexecution time: {}'.format(timedelta(seconds=round(elapsed_time_secs))))

    # for hp parameter tuning in TensorBoard
    if FLAGS.is_hyperparameter_tuning:
        logging.info('setup hyperparameter tuning!')
        logging.info('standard hyperparameter tuning!')
