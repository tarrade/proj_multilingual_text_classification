import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
from absl import logging
from absl import flags
from absl import app
import re
import tensorflow as tf
tf.get_logger().propagate = False
from transformers import (
    BertTokenizer,
    TFBertModel,
    glue_convert_examples_to_features,
)
import preprocessing.preprocessing as pp
import model.tf_bert_classification.model as tf_bert
import utils.model_utils as mu

print(tf.__version__)
print(tf.keras.__version__)

FLAGS = flags.FLAGS

# Maximum length, becareful BERT max length is 512!
MAX_LENGTH = 128

# define default parameters
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64
EPOCHS = 1
STEP_EPOCH_TRAIN = 10
STEP_EPOCH_VALID = 1

# hyper parameter
learning_rate=3e-5
epsilon=1e-08

# number of classes
NUM_CLASSES =2

# config
n_steps_history=10

# parameters for the training
flags.DEFINE_float('learning_rate', learning_rate, 'learning rate')
flags.DEFINE_float('epsilon', epsilon, 'epsilon')
flags.DEFINE_integer('epochs', EPOCHS, 'The number of epochs to train')
flags.DEFINE_integer('steps_per_epoch_train', STEP_EPOCH_TRAIN, 'The number of steps per epoch to train')
flags.DEFINE_integer('batch_size_train', BATCH_SIZE_TRAIN, 'Batch size for training')
flags.DEFINE_integer('steps_per_epoch_eval', STEP_EPOCH_VALID, 'The number of steps per epoch to evaluate')
flags.DEFINE_integer('batch_size_eval', BATCH_SIZE_VALID, 'Batch size for evaluation')
flags.DEFINE_integer('num_classes', NUM_CLASSES, 'number of classes in our model')
flags.DEFINE_integer('n_steps_history', n_steps_history, 'number of step for which we want custom history')
flags.DEFINE_string('input_train_tfrecords', '', 'input folder of tfrecords training data')
flags.DEFINE_string('input_eval_tfrecords', '', 'input folder of tfrecords evaluation data')
flags.DEFINE_string('output_dir', '', 'gs blob where are stored all the output of the model')
flags.DEFINE_string('pretrained_model_dir', '', 'number of classes in our model')
flags.DEFINE_enum('verbosity_level', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'], 'verbosity in the logfile')

print(list(FLAGS))
#print('Flags: \n',  FLAGS)

def main(argv):

    # set level of verbosity
    logging.set_verbosity(FLAGS.verbosity)

    # choose language's model and tokenizer
    MODELS = [(TFBertModel, BertTokenizer, 'bert-base-multilingual-uncased')]
    model_index = 0  # BERT
    model_class = MODELS[model_index][0]  # i.e TFBertModel
    tokenizer_class = MODELS[model_index][1]  # i.e BertTokenizer
    pretrained_weights = MODELS[model_index][2]  # 'i.e bert-base-multilingual-uncased'

    # download   pre trained model:
    if FLAGS.pretrained_model_dir:
        # download pre trained model from a bucket
        print('downloading pretrained model!')
        search = re.search('gs://(.*?)/(.*)', FLAGS.pretrained_model_dir)
        if search is not None:
            bucket_name = search.group(1)
            blob_name = search.group(2)
            local_path='.'
            mu.download_blob(bucket_name, blob_name, local_path)
            pretrained_model_dir = local_path+'/'+blob_name
        else:
            pretrained_model_dir = FLAGS.pretrained_model_dir
    else:
        # download pre trained model from internet
        pretrained_model_dir = '.'

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
        print('pretrained_model_dir=',pretrained_model_dir)
        model = tf_bert.create_model(pretrained_weights,
                                     pretrained_model_dir=pretrained_model_dir,
                                     num_labels=FLAGS.num_classes,
                                     learning_rate=FLAGS.learning_rate,
                                     epsilon=FLAGS.epsilon)

        tf_bert.train_and_evaluate(model,
                                   num_epochs=FLAGS.epochs,
                                   steps_per_epoch=FLAGS.steps_per_epoch_train,
                                   train_data=train_dataset,
                                   validation_steps=FLAGS.steps_per_epoch_eval,
                                   eval_data=valid_dataset,
                                   output_dir=FLAGS.output_dir,
                                   n_steps_history=FLAGS.n_steps_history)

if __name__ == '__main__':
    app.run(main)