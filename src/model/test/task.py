# import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
from transformers import BertTokenizer
import tensorflow as tf
from absl import logging
from absl import flags
from absl import app
import logging as logger
import google.cloud.logging

import sys

FLAGS = flags.FLAGS
flags.DEFINE_enum('verbosity_level', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'], 'verbosity in the logfile')


def main(argv):

    # set level of verbosity
    if FLAGS.verbosity_level == 'DEBUG':
        logging.set_verbosity(logging.DEBUG)
        print('logging.DEBUG')
    elif FLAGS.verbosity_level == 'INFO':
        logging.set_verbosity(logging.INFO)
        print('logging.INFO')
    elif FLAGS.verbosity_level == 'WARNING':
        logging.set_verbosity(logging.WARNING)
        print('logging.WARNING')
    elif FLAGS.verbosity_level == 'ERROR':
        logging.set_verbosity(logging.ERROR)
        print('logging.ERROR')
    elif FLAGS.verbosity_level == 'FATAL':
        logging.set_verbosity(logging.FATAL)
        print('logging.FATAL')
    else:
        logging.set_verbosity(logging.INFO)
        print('logging.DEFAULT -> INFO')

    # logging.get_absl_handler().python_handler.stream = sys.stdout

    # Instantiates a client
    client = google.cloud.logging.Client()

    # Connects the logger to the root logging handler; by default this captures
    # all logs at INFO level and higher
    client.setup_logging()

    fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    formatter = logger.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)

    # set level of verbosity
    # logging.set_verbosity(logging.DEBUG)

    # logging.set_stderrthreshold(logging.WARNING)
    # logging._warn_preinit_stderr = True
    # loggers = [logger.getLogger()]  # get the root logger

    # for handler in loggers:
    #    print("handler ", handler)
    #    print("       handler.level-->  ", handler.level)
    #    print("       handler.name-->  ", handler.name)
    #    print("       handler.propagate-->  ", handler.propagate)
    #    print("       handler.parent-->  ", handler.parent )
    #    print(dir(handler))
    # level_log = 'INFO'
    # root_logger = logger.getLogger()
    # root_logger.handlers=[handler for handler in root_logger.handlers if isinstance(handler, (CloudLoggingHandler, ContainerEngineHandler, logging.ABSLHandler))]
    #
    # for handler in root_logger.handlers:
    #    print("----- handler ", handler)
    #    print("---------class ", handler.__class__)
    #    if handler.__class__ == logging.ABSLHandler:
    #        handler.python_handler.stream = sys.stdout
    #        #handler.handler.setStream(sys.stdout)
    tf.get_logger().propagate = False
    root_logger = logger.getLogger()
    print(' root_logger :', root_logger)
    print(' root_logger.handlers :', root_logger.handlers)
    print(' len(root_logger) :', len(root_logger.handlers))
    for h in root_logger.handlers:
        print('handlers:', h)
        print("---------class ", h.__class__)
        if h.__class__ == logging.ABSLHandler:
            print('++logging.ABSLHandler')
            h.python_handler.stream = sys.stdout
            h.setLevel(logger.INFO)
        if h.__class__ == google.cloud.logging.handlers.handlers.CloudLoggingHandler:
            print('++CloudLoggingHandler')
            h.setLevel(logger.CRITICAL)
            h.setStream(sys.stdout)
            logger.getLogger().addHandler(h)
        if h.__class__ == logger.StreamHandler:
            print('++logging.StreamHandler')
            h.setLevel(logger.CRITICAL)
            h.setStream(sys.stdout)
            logger.getLogger().addHandler(h)

    logging.set_stderrthreshold(logging.WARNING)
    # handler = client.get_default_handler()
    # print('hhh', handler)
    # logger.getLogger().setLevel(logger.INFO)
    # logger.getLogger().addHandler(handler)

    # handler = logger.StreamHandler(sys.stderr)
    # handler.setLevel(logger.CRITICAL)
    # logger.getLogger().addHandler(handler)

    # handler = logger.StreamHandler(sys.stdout)
    # handler.setLevel(logger.CRITICAL)
    # logger.getLogger().addHandler(handler)

    print(' 0 print --- ')
    logging.info(' 1 logging:')
    logging.info(' 2 logging:')

    print(' 3 print --- ')
    logging.debug(' 4 logging-test-debug')
    logging.info(' 5 logging-test-info')
    logging.warning(' 6 logging-test-warning')
    logging.error(' 7 logging test-error')
    print(' 8 print --- ')
    _ = BertTokenizer.from_pretrained('bert-base-uncased')
    print(' 9 print --- ')
    _ = tf.distribute.MirroredStrategy()
    print('10 print --- ')


if __name__ == '__main__':
    app.run(main)
