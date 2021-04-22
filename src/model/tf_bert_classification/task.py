import os
import json
import pkgutil
from cloud_tpu_client import Client
from transformers import (
    BertTokenizer,
    TFBertModel,
)
import model.tf_bert_classification.model as tf_bert
import utils.model_utils as mu
import re
import sys

# import google.cloud.logging
# from google.cloud.logging.handlers import CloudLoggingHandler, ContainerEngineHandler
from absl import logging
from absl import flags
from absl import app
import logging as logger

# default parameters for training the model
# compute and save accuracy and loss after N steps
N_STEPS_HISTORY = 10
# define default parameters
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 64
EPOCHS = 1
STEP_EPOCH_TRAIN = 10
STEP_EPOCH_VALID = 1
# hyper parameters
# adam parameters
LEARNING_RATE = 3e-5
EPSILON = 1e-08
# learning rate decay parameters
DECAY_LR = 0.95
DECAY_TYPE = 'exponential'
N_BATCH_DECAY = 2
# number of classes
NUM_CLASSES = 2
# BERT Maximum length, be be careful BERT max length is 512!
MAX_LENGTH = 512

# get parameters for the training
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', LEARNING_RATE, 'learning rate')
flags.DEFINE_float('decay_learning_rate', DECAY_LR, 'decay of the learning rate, e.g. 0.9')
flags.DEFINE_float('epsilon', EPSILON, 'epsilon')
flags.DEFINE_integer('epochs', EPOCHS, 'The number of epochs to train')
flags.DEFINE_integer('steps_per_epoch_train', STEP_EPOCH_TRAIN, 'The number of steps per epoch to train')
flags.DEFINE_integer('batch_size_train', BATCH_SIZE_TRAIN, 'Batch size for training')
flags.DEFINE_integer('steps_per_epoch_eval', STEP_EPOCH_VALID, 'The number of steps per epoch to evaluate')
flags.DEFINE_integer('batch_size_eval', BATCH_SIZE_VALID, 'Batch size for evaluation')
flags.DEFINE_integer('num_classes', NUM_CLASSES, 'number of classes in our model')
flags.DEFINE_integer('n_steps_history', N_STEPS_HISTORY, 'number of step for which we want custom history')
flags.DEFINE_integer('n_batch_decay', N_BATCH_DECAY, 'number of batches after which the learning rate gets update')
flags.DEFINE_string('decay_type', DECAY_TYPE, 'type of decay for the learning rate: exponential, stepwise, timebased, or constant')
flags.DEFINE_string('input_train_tfrecords', None, 'input folder of tfrecords training data')
flags.DEFINE_string('input_eval_tfrecords', None, 'input folder of tfrecords evaluation data')
flags.DEFINE_string('output_dir', None, 'gs blob where are stored all the output of the model')
flags.DEFINE_string('pretrained_model_dir', None, 'number of classes in our model')
flags.DEFINE_enum('verbosity_level', 'INFO', ['VERBOSE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'], 'verbosity in the logfile')
flags.DEFINE_boolean('use_tpu', False, 'activate TPU for training')
flags.DEFINE_boolean('use_decay_learning_rate', False, 'activate decay learning rate')
flags.DEFINE_boolean('is_hyperparameter_tuning', False, 'automatic and inter flag')

# mandatory flags, for the other use default values
flags.mark_flag_as_required('input_train_tfrecords')
flags.mark_flag_as_required('input_eval_tfrecords')
flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('pretrained_model_dir')

# setup env variable for Tensorflow training before importing Tensorflow
json_data = pkgutil.get_data('utils', 'env_variables.json')
if json_data is not None:
    env_var = json.loads(json_data.decode('utf-8'))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(env_var['TF_CPP_MIN_LOG_LEVEL'])
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = str(env_var['TF_CPP_MIN_VLOG_LEVEL'])


def main(argv):
    import tensorflow as tf  # need to be here to have the env variables defined
    tf.get_logger().propagate = False

    # set level of verbosity
    if FLAGS.verbosity_level == 'DEBUG':
        logging.set_verbosity(logging.DEBUG)
        print('logging.DEBUG')
    elif FLAGS.verbosity_level == 'INFO':
        logging.set_verbosity(logging.INFO)
    elif FLAGS.verbosity_level == 'WARNING':
        logging.set_verbosity(logging.WARNING)
    elif FLAGS.verbosity_level == 'ERROR':
        logging.set_verbosity(logging.ERROR)
    elif FLAGS.verbosity_level == 'FATAL':
        logging.set_verbosity(logging.FATAL)
    else:
        logging.set_verbosity(logging.INFO)

    # set level of verbosity for Tensorflow
    if FLAGS.verbosity_level == 'VERBOSE':
        tf.debugging.set_log_device_placement(True)
        tf.autograph.set_verbosity(10, alsologtostdout=False)

    # logger.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

    # fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    fmt = "[%(levelname)s] %(message)s"
    formatter = logger.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_stderrthreshold(logging.WARNING)

    # level_log = 'INFO'

    # # Instantiates a client
    # client = google.cloud.logging.Client()
    #
    # # Connects the logger to the root logging handler; by default this captures
    # # all logs at INFO level and higher
    # client.setup_logging(log_level=FLAGS.verbosity)
    #
    # print('loggerDict:', logger.root.manager.loggerDict.keys())
    #
    # for i in logger.root.manager.loggerDict.keys():
    #     if i=='tensorflow':
    #        #print('-> propagate False')
    #         logger.getLogger(i).propagate = False  # needed
    #     elif i=='google.auth':
    #         logger.getLogger(i).propagate = False  # needed
    #     elif i=='google_auth_httplib2':
    #         logger.getLogger(i).propagate = False  # needed
    #     elif i=='pyasn1':
    #         logger.getLogger(i).propagate = False  # needed
    #     elif i=='sklearn':
    #         logger.getLogger(i).propagate = False  # needed
    #     elif i=='google.cloud':
    #         logger.getLogger(i).propagate = False  # needed
    #     else:
    #         logger.getLogger(i).propagate = True # needed
    #     handler = logger.getLogger(i).handlers
    #     if handler != []:
    #         #print("logger's name=", i,handler)
    #         for h in handler:
    #             #print('    -> ', h)
    #             if h.__class__ == logger.StreamHandler:
    #                 #print('    -> name=', h.__class__)
    #                 h.setStream(sys.stdout)
    #                 h.setLevel(level_log)
    #                 #print('    --> handlers =', h)
    #
    root_logger = logger.getLogger()
    # root_logger.handlers=[handler for handler in root_logger.handlers if isinstance(handler, (CloudLoggingHandler, ContainerEngineHandler, logging.ABSLHandler))]
    #
    for handler in root_logger.handlers:
        print("----- handler ", handler)
        print("---------class ", handler.__class__)

    #     if handler.__class__ == CloudLoggingHandler:
    #         handler.setStream(sys.stdout)
    #         handler.setLevel(level_log)
    #     if handler.__class__ == logging.ABSLHandler:
    #         handler.python_handler.stream = sys.stdout
    #         handler.setLevel(level_log)
    # #        handler.handler.setStream(sys.stdout)
    #
    # for handler in root_logger.handlers:
    #     print("----- handler ", handler)
    #
    # # Instantiates a client
    # #client = google.cloud.logging.Client()
    #
    # # Connects the logger to the root logging handler; by default this captures
    # # all logs at INFO level and higher
    # #client.setup_logging()
    #
    # # redirect abseil logging messages to the stdout stream
    # #logging.get_absl_handler().python_handler.stream = sys.stdout
    #
    # # some test
    # #tf.get_logger().addHandler(logger.StreamHandler(sys.stdout))
    # #tf.get_logger().disabled = True
    # #tf.autograph.set_verbosity(5 ,alsologtostdout=True)
    #
    # ## DEBUG
    # #fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    # #formatter = logger.Formatter(fmt)
    # #logging.get_absl_handler().setFormatter(formatter)
    #
    # # set level of verbosity
    # #logging.set_verbosity(logging.DEBUG)
    #
    # print(' 0 print --- ')
    # logging.info(' 1 logging:')
    # logging.info(' 2 logging:')
    #
    # print(' 3 print --- ')
    # logging.debug(' 4 logging-test-debug')
    # logging.info(' 5 logging-test-info')
    # logging.warning(' 6 logging-test-warning')
    # logging.error(' 7 logging test-error')
    # print(' 8 print --- ')
    # #_=BertTokenizer.from_pretrained('bert-base-uncased')
    # print(' 9 print --- ')
    # _= tf.distribute.MirroredStrategy()
    # print('10 print --- ')
    # ## DEBUG

    print('logging.get_verbosity()', logging.get_verbosity())

    # print flags
    abseil_flags = ['logtostderr', 'alsologtostderr', 'log_dir', 'v', 'verbosity', 'stderrthreshold',
                    'showprefixforinfo', 'run_with_pdb', 'pdb_post_mortem', 'run_with_profiling', 'profile_file',
                    'use_cprofile_for_profiling', 'only_check_args', 'flagfile', 'undefok']
    logging.info('-- Custom flags:')
    for name in list(FLAGS):
        if name not in abseil_flags:
            logging.info('custom flags: {:40} with value: {:50}'.format(name, str(FLAGS[name].value)))
    logging.info('\n-- Abseil flags:')
    for name in list(FLAGS):
        if name in abseil_flags:
            logging.info('abseil flags: {:40} with value: {:50}'.format(name, str(FLAGS[name].value)))

    if os.environ.get('LOG_FILE_TO_WRITE') is not None:
        logging.info('os.environ[LOG_FILE_TO_WRITE]: {}'.format(os.environ['LOG_FILE_TO_WRITE']))
        # split_path = os.environ['LOG_FILE_TO_WRITE'].split('/')
        # logging.get_absl_handler().use_absl_log_file(split_path[-1], '/'.join(split_path[:-1]))

    # fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    # formatter = logger.Formatter(fmt)
    # logging.get_absl_handler().setFormatter(formatter)

    # set level of verbosity
    # logging.set_verbosity(FLAGS.verbosity)
    # logging.set_stderrthreshold(FLAGS.verbosity)

    logging.info(tf.__version__)
    logging.info(tf.keras.__version__)
    logging.info(list(FLAGS))
    logging.debug('flags: \n {}'.format(FLAGS))
    logging.debug('env variables: \n{}'.format(os.environ))
    logging.debug('current dir: {}'.format(os.getcwd()))
    logging.debug('__package__: {}'.format(__package__))
    logging.debug('__name__: {}'.format(__name__))
    logging.debug('__file__: {}'.format(__file__))

    # only for HP tuning!
    if os.environ.get('CLOUD_ML_HP_METRIC_TAG') is not None:
        logging.info('this is a hyper parameters job !')

        # setup the hp flag
        FLAGS.is_hyperparameter_tuning = True
        logging.info('FLAGS.is_hyperparameter_tuning: {}'.format(FLAGS.is_hyperparameter_tuning))

        logging.info('os.environ[CLOUD_ML_HP_METRIC_TAG]: {}'.format(os.environ['CLOUD_ML_HP_METRIC_TAG']))
        logging.info('os.environ[CLOUD_ML_HP_METRIC_FILE]: {}'.format(os.environ['CLOUD_ML_HP_METRIC_FILE']))
        logging.info('os.environ[CLOUD_ML_TRIAL_ID]: {}'.format(os.environ['CLOUD_ML_TRIAL_ID']))

        # variable name for hyper parameter tuning
        metric_accuracy = os.environ['CLOUD_ML_HP_METRIC_TAG']
        logging.info('metric accuracy name: {}'.format(metric_accuracy))
    else:
        metric_accuracy = 'NotDefined'

    if os.environ.get('TF_CONFIG') is not None:
        logging.info('os.environ[TF_CONFIG]: {}'.format(os.environ['TF_CONFIG']))
    else:
        logging.error('os.environ[TF_CONFIG] doesn\'t exist !')

    if os.environ.get('CLUSTER_SPEC') is not None:
        logging.info('os.environ[CLUSTER_SPEC]: {}'.format(os.environ['CLUSTER_SPEC']))
    else:
        logging.error('os.environ[CLUSTER_SPEC] doesn\'t exist !')

    if FLAGS.use_tpu:
        # Check or update the TensorFlow on the TPU cluster to match the one of the VM
        logging.info('setting up TPU: check that TensorFlow version is the same on the VM and on the TPU cluster')
        client_tpu = Client()

        # define TPU strategy before any ops
        client_tpu.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
        logging.info('setting up TPU: cluster resolver')
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        logging.info('setting up TPU: \n {}'.format(tpu_cluster_resolver))
        logging.info('running on TPU: \n {}'.format(tpu_cluster_resolver.cluster_spec().as_dict()['worker']))
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
        print('do nothing')
    logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # choose language's model and tokenizer
    MODELS = [(TFBertModel, BertTokenizer, 'bert-base-multilingual-uncased')]
    model_index = 0  # BERT
    # model_class = MODELS[model_index][0]  # i.e TFBertModel
    # tokenizer_class = MODELS[model_index][1]  # i.e BertTokenizer
    pretrained_weights = MODELS[model_index][2]  # 'i.e bert-base-multilingual-uncased'

    # download pre trained model:
    if FLAGS.pretrained_model_dir:
        # download pre trained model from a bucket
        search = re.search('gs://(.*?)/(.*)', FLAGS.pretrained_model_dir)
        if search is not None:
            bucket_name = search.group(1)
            blob_name = search.group(2)
            local_path = '.'
            mu.download_blob(bucket_name, blob_name, local_path)
            pretrained_model_dir = local_path + '/' + blob_name
            logging.info('downloading pretrained model from gcs and stored in {}'.format(pretrained_model_dir))
        else:
            pretrained_model_dir = FLAGS.pretrained_model_dir
            logging.info('use pretrained model from {}'.format(pretrained_model_dir))
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

    # with tf.summary.create_file_writer(FLAGS.output_dir,
    #                                   filename_suffix='.oup',
    #                                   name='test').as_default():
    #    tf.summary.scalar('metric_accuracy', 1.0, step=1)
    # print('-- 00001')
    #  read TFRecords files, shuffle, map and batch size
    train_dataset = tf_bert.build_dataset(FLAGS.input_train_tfrecords, FLAGS.batch_size_train, 2048)
    valid_dataset = tf_bert.build_dataset(FLAGS.input_eval_tfrecords, FLAGS.batch_size_eval, 2048)

    # set repeat
    train_dataset = train_dataset.repeat(FLAGS.epochs + 1)
    valid_dataset = valid_dataset.repeat(2)

    for i in train_dataset:
        logging.info('dataset training ={}'.format(FLAGS.input_train_tfrecords))
        logging.info('batch size training ={}'.format(FLAGS.batch_size_train))
        logging.info('nb epoch training ={}'.format(FLAGS.epochs + 1))
        logging.info('shape data training input_ids ={}'.format(i[0]['input_ids'].shape))
        break

    # reset all variables used by Keras
    tf.keras.backend.clear_session()

    # create and compile the Keras model in the context of strategy.scope
    with strategy.scope():
        logging.debug('pretrained_model_dir={}'.format(pretrained_model_dir))
        model = tf_bert.create_model(pretrained_weights,
                                     pretrained_model_dir=pretrained_model_dir,
                                     num_labels=FLAGS.num_classes,
                                     learning_rate=FLAGS.learning_rate,
                                     epsilon=FLAGS.epsilon)
    # train the model
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
                               s=FLAGS.decay_learning_rate,
                               n_batch_decay=FLAGS.n_batch_decay,
                               metric_accuracy=metric_accuracy)


if __name__ == '__main__':
    app.run(main)
