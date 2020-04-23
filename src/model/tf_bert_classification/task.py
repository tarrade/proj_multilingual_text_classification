import argparse
import sys

import tensorflow as tf

def _parse_arguments(argv):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        help='The number of epochs to train',
        type=int, default=5)
    parser.add_argument(
        '--steps_per_epoch',
        help='The number of steps per epoch to train',
        type=int, default=500)
    parser.add_argument(
        '--train_path',
        help='The path to the training data',
        type=str, default="gs://cloud-ml-data/img/flower_photos/train_set.csv")
    parser.add_argument(
        '--eval_path',
        help='The path to the evaluation data',
        type=str, default="gs://cloud-ml-data/img/flower_photos/eval_set.csv")
    parser.add_argument(
        '--tpu_address',
        help='The path to the evaluation data',
        type=str, required=True)
    parser.add_argument(
        '--hub_path',
        help='The path to TF Hub module to use in GCS',
        type=str, required=True)
    parser.add_argument(
        '--job-dir',
        help='Directory where to save the given model',
        type=str, required=True)
    return parser.parse_known_args(argv)


def main():
    """Parses command line arguments and kicks off model training."""

    MODELS = [(TFBertModel, BertTokenizer, 'bert-base-multilingual-uncased'),
              (TFXLMRobertaModel, XLMRobertaTokenizer, 'jplu/tf-xlm-roberta-base')]
    model_index = 0  # BERT
    model_class = MODELS[model_index][0]  # i.e TFBertModel
    tokenizer_class = MODELS[model_index][1]  # i.e BertTokenizer
    pretrained_weights = MODELS[model_index][2]  # 'i.e bert-base-multilingual-uncased'

    # Maxium length, becarefull BERT max length is 512!
    MAX_LENGTH = 128

    # define parameters
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 32
    BATCH_SIZE_VALID = 64
    EPOCH = 2

    # extract parameters
    if tf.version.VERSION[0:5] == '2.2.0':
        size_train_dataset = tf.data.experimental.cardinality(train_dataset)
        size_test_dataset = tf.data.experimental.cardinality(test_dataset)
        size_valid_dataset = tf.data.experimental.cardinality(valid_dataset)
    else:
        size_train_dataset = train_dataset.reduce(0, lambda x, _: x + 1).numpy()
        size_test_dataset = test_dataset.reduce(0, lambda x, _: x + 1).numpy()
        size_valid_dataset = valid_dataset.reduce(0, lambda x, _: x + 1).numpy()
    number_label = 2

    # computer parameter
    STEP_EPOCH_TRAIN = math.ceil(size_train_dataset / BATCH_SIZE_TRAIN)
    STEP_EPOCH_TEST = math.ceil(size_test_dataset / BATCH_SIZE_TEST)
    STEP_EPOCH_VALID = math.ceil(size_test_dataset / BATCH_SIZE_VALID)


    args = _parse_arguments(sys.argv[1:])[0]

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # create and compile the Keras model in the context of strategy.scope
    with strategy.scope():
        model = tf_bert.create_model(pretrained_weights,
                                     pretrained_model_dir=pretrained_model_dir,
                                     num_labels=number_label,
                                     learning_rate=3e-5,
                                     epsilon=1e-08)

    model_history = model.train_and_evaluate(
        image_model, args.epochs, args.steps_per_epoch,
        train_data, eval_data, args.job_dir)


if __name__ == '__main__':
    main()