import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
import numpy as np
import pprint

# function to print details/info about glue tensorflow dataset
def print_info_dataset(info):
    print('Labels:\n      {}\n'.format(info.features["label"].names))
    print('Number of label:\n      {}\n'.format(info.features["label"].num_classes))
    print('Structure of the data:\n      {}\n'.format(info.features.keys()))
    print('Number of entries:')
    try:
        print('   Train dataset: {}'.format(info.splits['train'].num_examples))
    except ValueError:
        print('--> train dataset not defined')
    try:
        print('   Test dataset:  {}'.format(info.splits['test'].num_examples))
    except ValueError:
        print('--> test dataset not defined')
    try:
        print('   Valid dataset: {}\n'.format(info.splits['validation'].num_examples))
    except ValueError:
        print('--> validation dataset not defined')

# function to print data structure/shape about glue tensorflow dataset
def print_info_data(dataset, print_example=True, n_example=3):
    print('# Structure of the data:\n\n   {}'.format(dataset))
    print('\n# Output shape of one entry:\n   {}'.format(dataset_ops.get_legacy_output_shapes(dataset)))
    print('\n# Output types of one entry:\n   {}'.format(dataset_ops.get_legacy_output_types(dataset)))
    print('\n# Output typesof one entry:\n   {}'.format(dataset_ops.get_legacy_output_classes(dataset)))
    print(' \n')
    np_array = np.array(list(dataset.as_numpy_iterator()))
    print('# Shape of the data:\n\n   {}'.format(np.shape(np_array)))
    if len(np_array) > 0:
        if type(np_array[0]) is dict:
            structure = list(np_array[0].keys())
            print('   ---> {} entries'.format(np.shape(np_array)[0]))
            print('   ---> {} dim'.format(np_array.ndim))
            print('        dict structure')
            print('           dim: {}'.format(len(structure)))
            print('           [{:9} / {:9} / {:9}]'.format(structure[0], structure[1], structure[2]))

            print('           [{:9} / {:9} / {:9}]'.format(str(np.shape(np_array[0].get(structure[0]))),
                                                           str(np.shape(np_array[0].get(structure[1]))),
                                                           str(np.shape(np_array[0].get(structure[2])))))
            print('           [{:9} / {:9} / {:9}]'.format(type(np_array[0].get(structure[0])).__name__,
                                                           type(np_array[0].get(structure[1])).__name__,
                                                           type(np_array[0].get(structure[2])).__name__))

        if type(np_array[0]) is np.ndarray:
            structure = list(np_array[0][0].keys())
            print('   ---> {} batches'.format(np.shape(np_array)[0]))
            print('   ---> {} dim'.format(np_array.ndim))
            print('        label')
            print('           shape: {}'.format(np_array[0][1].shape))
            print('        dict structure')
            print('           dim: {}'.format(len(structure)))
            print('           [{:15} / {:15} / {:15}]'.format(structure[0], structure[1], structure[2]))
            print('           [{:15} / {:15} / {:15}]'.format(str(np_array[0][0].get(structure[0]).shape),
                                                              str(np_array[0][0].get(structure[1]).shape),
                                                              str(np_array[0][0].get(structure[2]).shape)))
            print('           [{:15} / {:15} / {:15}]'.format(type(np_array[0][0].get(structure[0])).__name__,
                                                              type(np_array[0][0].get(structure[1])).__name__,
                                                              type(np_array[0][0].get(structure[2])).__name__))

    if print_example:
        print('\n\n# Examples of data:')
        for i, ex in enumerate(np_array):
            print('{}'.format(pprint.pformat(ex)))
            if i + 1 > n_example:
                break

# print details on one example of the tokenize data
def print_detail_tokeniser(dataset, tokenizer, max_entries=20):
    np_array = np.array(list(dataset.take(1).as_numpy_iterator()))
    print('{:>10}     ---->    {:^15}   {:^15}   {:<30}\n'.format('input_ids', 'attention_mask', 'token_type_ids',
                                                                  'modified text'))
    for i, v in enumerate(np_array[0][0]['input_ids'][0]):
        print('{:>10}     ---->    {:^15d}   {:^15d}   {:<30}'.format(v,
                                                                      int(np_array[0][0]['attention_mask'][0][i]),
                                                                      int(np_array[0][0]['attention_mask'][0][i]),
                                                                      tokenizer.decode(int(v))))
        if i > max_entries:
            break

# create a data structure
class InputFeatures(object):
    def __init__(self, idx, label,sentence):
        self.idx = idx
        self.sentence = sentence
        self.label = label

# define an iterable
def gen(features):
    for f in features:
        yield ({"idx": f.idx,
                "label": f.label,
                "sentence": f.sentence,
                })

# transofrm and return data in the right structure
def create_tf_example(idx, label, sentence):

    features = []
    for i, x in enumerate(sentence):
        features.append(InputFeatures(np.int32(idx[i]), np.int64(label[i]), sentence[i]))

    return tf.data.Dataset.from_generator(
        lambda: gen(features),
        ({"idx": tf.int32, "label": tf.int64, "sentence": tf.string}),
        ({"idx": tf.TensorShape([]),
          "label": tf.TensorShape([]),
          "sentence": tf.TensorShape([]),
          })
    )

