import argparse
import sys
from absl import logging
from absl import flags
from absl import app
import tensorflow as tf
from transformers import (
    BertTokenizer,
    TFBertModel,
    glue_convert_examples_to_features,
)
import preprocessing.preprocessing as pp
import model.tf_bert_classification.model as tf_bert

FLAGS = flags.FLAGS

# Maximum length, becareful BERT max length is 512!
MAX_LENGTH = 128

# define default parameters
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
BATCH_SIZE_VALID = 64
EPOCHS = 1
STEP_EPOCH_TRAIN = 10
STEP_EPOCH_VALID = 1

# number of classes
NUM_CLASSES =2

# parameters for the training
flags.DEFINE_string('input_train_tfrecords', '', 'input folder of tfrecords training data')
flags.DEFINE_string('input_eval_tfrecords', '', 'input folder of tfrecords evaluation data')
flags.DEFINE_integer('epochs', EPOCHS, 'The number of epochs to train')
flags.DEFINE_integer('steps_per_epoch_train', STEP_EPOCH_TRAIN, 'The number of steps per epoch to train')
flags.DEFINE_integer('batch_size_train', BATCH_SIZE_TRAIN, 'Batch size for training')
flags.DEFINE_integer('steps_per_epoch_eval', STEP_EPOCH_TRAIN, 'The number of steps per epoch to evaluate')
flags.DEFINE_integer('batch_size_eval', BATCH_SIZE_TRAIN, 'Batch size for evaluation')
flags.DEFINE_integer('num_classes', NUM_CLASSES, 'number of classes in our model')
flags.DEFINE_string('output_dir', '', 'number of classes in our model')
flags.DEFINE_string('job-dir', '', 'number of classes in our model')
flags.DEFINE_string('pretrained_model_dir', '', 'number of classes in our model')
flags.DEFINE_enum('verbosity_level', 'INFO', ['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'], 'verbosity in the logfile')


def main(argv):

    # set level of verbosity
    logging.set_verbosity(FLAGS.verbosity)

    # choose language's model and tokenizer
    MODELS = [(TFBertModel, BertTokenizer, 'bert-base-multilingual-uncased')]
    model_index = 0  # BERT
    model_class = MODELS[model_index][0]  # i.e TFBertModel
    tokenizer_class = MODELS[model_index][1]  # i.e BertTokenizer
    pretrained_weights = MODELS[model_index][2]  # 'i.e bert-base-multilingual-uncased'

    # read TFRecords files
    train_files = tf.data.TFRecordDataset(FLAGS.input_train_tfrecords + '/train_dataset.tfrecord')
    valid_files = tf.data.TFRecordDataset(FLAGS.input_eval_tfrecords + '/valid_dataset.tfrecord')

    train_dataset = train_files.map(pp.parse_tfrecord_glue_files)
    valid_dataset = valid_files.map(pp.parse_tfrecord_glue_files)

    # set shuffle and batch size
    train_dataset = train_dataset.shuffle(100).batch(FLAGS.batch_size_train).repeat(FLAGS.epochs + 1)
    valid_dataset = valid_dataset.batch(FLAGS.batch_size_eval)

    # reset Keras
    tf.keras.backend.clear_session()

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # create and compile the Keras model in the context of strategy.scope
    with strategy.scope():
        model = tf_bert.create_model(pretrained_weights,
                                     pretrained_model_dir=FLAGS.pretrained_model_dir,
                                     num_labels=FLAGS.num_classes,
                                     learning_rate=3e-5,
                                     epsilon=1e-08)

    model_history = tf_bert.train_and_evaluate(model,
                                               num_epochs=FLAGS.epochs,
                                               steps_per_epoch=FLAGS.steps_per_epoch_train,
                                               train_data=train_dataset,
                                               validation_steps=FLAGS.steps_per_epoch_eval,
                                               eval_data=valid_dataset,
                                               output_dir=FLAGS.output_dir)

if __name__ == '__main__':
    app.run(main)