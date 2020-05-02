import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification,
)
import utils.model_utils as mu
import os
import glob
import re
import pickle
from absl import logging

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


def train_and_evaluate(model, num_epochs, steps_per_epoch, train_data, validation_steps, eval_data, output_dir, n_steps_history):
    """Compiles keras model and loads data into it for training."""
    logging.info('training the model ...')
    model_callbacks = []

    if output_dir:
        # tensorflow callback
        log_dir = os.path.join(output_dir, 'tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              embeddings_freq=1,
                                                              write_graph=True,
                                                              update_freq='batch',
                                                              profile_batch=1)
        model_callbacks.append(tensorboard_callback)

        # checkpoints callback
        checkpoint_dir = os.path.join(output_dir, 'checkpoint_model')
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

    # train the model
    history = model.fit(train_data,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=eval_data,
                        validation_steps=validation_steps,
                        callbacks=model_callbacks)

    # save the history in a file
    history_dir = os.path.join('./', model.name)
    os.makedirs(history_dir, exist_ok=True)
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
        model.save(os.path.join(savemodel_path, model.name))

        # save history
        search = re.search('gs://(.*?)/(.*)', output_dir)
        if search is not None:
            bucket_name = search.group(1)
            blob_name = search.group(2)
            mu.copy_local_directory_to_gcs(history_dir, bucket_name, blob_name+'/history')

