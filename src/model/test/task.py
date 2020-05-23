import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
import tensorflow as tf
tf.get_logger().propagate = False
from transformers import (
    BertTokenizer,
    TFBertModel,
    glue_convert_examples_to_features,
)
from absl import logging
from absl import flags
from absl import app
import logging as logger
import google.cloud.logging

import sys

FLAGS = flags.FLAGS
flags.DEFINE_enum('verbosity_level', 'INFO', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'], 'verbosity in the logfile')

def main(argv):

    logging.get_absl_handler().python_handler.stream = sys.stdout

    # Instantiates a client
    client = google.cloud.logging.Client()

    # Connects the logger to the root logging handler; by default this captures
    # all logs at INFO level and higher
    #client.setup_logging()

    fmt = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    formatter = logger.Formatter(fmt)
    logging.get_absl_handler().setFormatter(formatter)

    # set level of verbosity
    logging.set_verbosity(logging.DEBUG)

    logging.set_stderrthreshold(logging.WARNING)
    logging._warn_preinit_stderr = False


    loggers = [logger.getLogger()]  # get the root logger

    #for handler in loggers:
    #    print("handler ", handler)
    #    print("       handler.level-->  ", handler.level)
    #    print("       handler.name-->  ", handler.name)
    #    print("       handler.propagate-->  ", handler.propagate)
    #    print("       handler.parent-->  ", handler.parent )
    #    print(dir(handler))

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