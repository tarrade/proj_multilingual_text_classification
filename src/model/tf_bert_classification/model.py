import tensorflow as tf
from transformers import (
    TFBertForSequenceClassification,
)

# to be removed
import glob

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

    print('pretrained model\'s files: \n', glob.glob(pretrained_model_dir+"/*"))

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


def train_and_evaluate(model, num_epochs, steps_per_epoch, train_data, validation_steps, eval_data, output_dir):
    """Compiles keras model and loads data into it for training."""

    model_callbacks = []
    if output_dir:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=output_dir,
                                                              histogram_freq=1,
                                                              embeddings_freq=1,
                                                              write_graph=True,
                                                              update_freq='batch',
                                                              profile_batch=1)
        model_callbacks.append(tensorboard_callback)

    # train the model
    history = model.fit(train_data,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=eval_data,
                        validation_steps=validation_steps,
                        callbacks=model_callbacks)

    #if output_dir:
    #    export_path = os.path.join(output_dir, 'keras_export')
    #    model.save(export_path, save_format='tf')

    return history