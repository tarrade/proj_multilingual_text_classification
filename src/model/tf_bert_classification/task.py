import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
import tensorflow as tf
tf.get_logger().propagate = False
from absl import logging
from absl import flags
from absl import app
import logging as logger
from transformers import (
    BertTokenizer,
    TFBertModel,
    glue_convert_examples_to_features,
)
import model.tf_bert_classification.model as tf_bert
import utils.model_utils as mu
import re
import sys
import google.cloud.logging

FLAGS = flags.FLAGS


# Maximum length, be becareful BERT max length is 512!
MAX_LENGTH = 512

# define default parameters
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64
EPOCHS = 1
STEP_EPOCH_TRAIN = 10
STEP_EPOCH_VALID = 1

# hyper parameter
learning_rate=3e-5
epsilon=1e-08
s=0.95
decay_type='exponential'
n_batch_decay=2

# number of classes
NUM_CLASSES =2

# config
n_steps_history=10

# parameters for the training
flags.DEFINE_float('learning_rate', learning_rate, 'learning rate')
flags.DEFINE_float('s', s, 'decay of the learning rate, e.g. 0.9')
flags.DEFINE_float('epsilon', epsilon, 'epsilon')
flags.DEFINE_integer('epochs', EPOCHS, 'The number of epochs to train')
flags.DEFINE_integer('steps_per_epoch_train', STEP_EPOCH_TRAIN, 'The number of steps per epoch to train')
flags.DEFINE_integer('batch_size_train', BATCH_SIZE_TRAIN, 'Batch size for training')
flags.DEFINE_integer('steps_per_epoch_eval', STEP_EPOCH_VALID, 'The number of steps per epoch to evaluate')
flags.DEFINE_integer('batch_size_eval', BATCH_SIZE_VALID, 'Batch size for evaluation')
flags.DEFINE_integer('num_classes', NUM_CLASSES, 'number of classes in our model')
flags.DEFINE_integer('n_steps_history', n_steps_history, 'number of step for which we want custom history')
flags.DEFINE_integer('n_batch_decay', n_batch_decay, 'number of batches after which the learning rate gets update')
flags.DEFINE_string('decay_type', decay_type, 'type of decay for the learning rate: exponential, stepwise, timebased, or constant')
flags.DEFINE_string('input_train_tfrecords', '', 'input folder of tfrecords training data')
flags.DEFINE_string('input_eval_tfrecords', '', 'input folder of tfrecords evaluation data')
flags.DEFINE_string('output_dir', '', 'gs blob where are stored all the output of the model')
flags.DEFINE_string('pretrained_model_dir', '', 'number of classes in our model')
flags.DEFINE_enum('verbosity_level', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'], 'verbosity in the logfile')
flags.DEFINE_boolean('use_tpu', False, 'activate TPU for training')
flags.DEFINE_boolean('use_decay_learning_rate', False, 'activate decay learning rate')
flags.DEFINE_boolean('is_hyperparameter_tuning', False, 'automatic and inter flag')

def main(argv):

    # Instantiates a client
    client = google.cloud.logging.Client()

    # Connects the logger to the root logging handler; by default this captures
    # all logs at INFO level and higher
    client.setup_logging()

    logging.get_absl_handler().python_handler.stream = sys.stdout

    tf.get_logger().addHandler(logger.StreamHandler(sys.stdout))
    #tf.get_logger().disabled = True
    tf.autograph.set_verbosity(5 ,alsologtostdout=True)

    ## DEBUG
    fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    formatter = logger.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)

    # set level of verbosity
    logging.set_verbosity(logging.DEBUG)

    print(' 0 print --- ')
    logging.info(' 1 logging:'.format(tf.__version__))
    logging.info(' 2 logging:'.format(tf.__version__))

    print(' 3 print --- ')
    logging.debug(' 4 logging-test-debug')
    logging.info(' 5 logging-test-info')
    logging.warning(' 6 logging-test-warning')
    logging.error(' 7 logging test-error')
    print(' 8 print --- ')
    strategy = tf.distribute.MirroredStrategy()
    print(' 9 print --- ')
    print('loggerDict:', logger.root.manager.loggerDict.keys())
    print(' 10 print --- ')
    ## DEBUG

    if os.environ.get('LOG_FILE_TO_WRITE') is not None:
        logging.info('os.environ[LOG_FILE_TO_WRITE]: {}'.format(os.environ['LOG_FILE_TO_WRITE']))
        #split_path = os.environ['LOG_FILE_TO_WRITE'].split('/')
        #logging.get_absl_handler().use_absl_log_file(split_path[-1], '/'.join(split_path[:-1]))

    #fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    #formatter = logger.Formatter(fmt)
    #logging.get_absl_handler().setFormatter(formatter)

    # set level of verbosity
    logging.set_verbosity(FLAGS.verbosity)
    #logging.set_stderrthreshold(FLAGS.verbosity)

    logging.info(tf.__version__)
    logging.info(tf.keras.__version__)
    logging.info(list(FLAGS))
    logging.debug('flags: \n {}'.format(FLAGS))
    logging.debug('env variables: \n{}'.format(os.environ))

    # only for HP tuning!
    if os.environ.get('CLOUD_ML_HP_METRIC_TAG') is not None:
        logging.info('this is a hyper parameters job !')

        # setup the hp flag
        FLAGS.is_hyperparameter_tuning=True
        logging.info('FLAGS.is_hyperparameter_tuning: {}'.format(FLAGS.is_hyperparameter_tuning))

        logging.info('os.environ[CLOUD_ML_HP_METRIC_TAG]: {}'.format(os.environ['CLOUD_ML_HP_METRIC_TAG']))
        logging.info('os.environ[CLOUD_ML_HP_METRIC_FILE]: {}'.format(os.environ['CLOUD_ML_HP_METRIC_FILE']))
        logging.info('os.environ[CLOUD_ML_TRIAL_ID]: {}'.format(os.environ['CLOUD_ML_TRIAL_ID']))

    if os.environ.get('TF_CONFIG') is not None:
        logging.info('os.environ[TF_CONFIG]: {}'.format(os.environ['TF_CONFIG']))
    else:
        logging.error('os.environ[TF_CONFIG] doesn\'t exist !')

    # choose language's model and tokenizer
    MODELS = [(TFBertModel, BertTokenizer, 'bert-base-multilingual-uncased')]
    model_index = 0  # BERT
    model_class = MODELS[model_index][0]  # i.e TFBertModel
    tokenizer_class = MODELS[model_index][1]  # i.e BertTokenizer
    pretrained_weights = MODELS[model_index][2]  # 'i.e bert-base-multilingual-uncased'

    # download   pre trained model:
    if FLAGS.pretrained_model_dir:
        # download pre trained model from a bucket
        logging.info('downloading pretrained model!')
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

    # some check
    logging.info('Batch size:            {:6}/{:6}'.format(FLAGS.batch_size_train,
                                                           FLAGS.batch_size_eval))
    logging.info('Step per epoch:        {:6}/{:6}'.format(FLAGS.steps_per_epoch_train,
                                                           FLAGS.steps_per_epoch_eval))
    logging.info('Total number of batch: {:6}/{:6}'.format(FLAGS.steps_per_epoch_train * (FLAGS.epochs + 1),
                                                           FLAGS.steps_per_epoch_eval * 1))
    # read TFRecords files
    #train_files = tf.io.gfile.glob(FLAGS.input_train_tfrecords+'/*.tfrecord')
    #valid_files = tf.io.gfile.glob(FLAGS.input_eval_tfrecords+'/*.tfrecord')
    #train_dataset = tf_bert.build_dataset(train_files, FLAGS.batch_size_train, 2048)
    #valid_dataset = tf_bert.build_dataset(valid_files, FLAGS.batch_size_eval, 2048)
    #print('test 1:', list(tf.data.Dataset.list_files(tf.io.gfile.glob(FLAGS.input_train_tfrecords+'/*.tfrecord'))))

    #  set shuffle, map and batch size
    train_dataset = tf_bert.build_dataset(FLAGS.input_train_tfrecords, FLAGS.batch_size_train, 2048)
    valid_dataset = tf_bert.build_dataset(FLAGS.input_eval_tfrecords, FLAGS.batch_size_eval, 2048)

    # set repeat
    train_dataset = train_dataset.repeat(FLAGS.epochs + 1)
    valid_dataset = valid_dataset.repeat(2)

    # reset Keras
    tf.keras.backend.clear_session()

    if FLAGS.use_tpu:
        logging.info('setting up TPU: cluster resolver')
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        logging.info('setting up TPU: \n {}'.format(tpu_cluster_resolver))
        logging.info('running on TPU: \n {}'.format(tpu_cluster_resolver.cluster_spec().as_dict()['worker']))
        #tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        #    FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
        #    zone=FLAGS.tpu_zone,
        #    project=FLAGS.gcp_project)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # create and compile the Keras model in the context of strategy.scope
    with strategy.scope():
        logging.debug('pretrained_model_dir={}'.format(pretrained_model_dir))
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
                                   n_steps_history=FLAGS.n_steps_history,
                                   FLAGS=FLAGS,
                                   decay_type=FLAGS.decay_type,
                                   learning_rate=FLAGS.learning_rate,
                                   s=FLAGS.s,
                                   n_batch_decay=FLAGS.n_batch_decay)

if __name__ == '__main__':
    app.run(main)