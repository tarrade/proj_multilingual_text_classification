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
            if type(np_array[0][0]) is dict:
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
            else:
                print('   ---> {} entries'.format(np.shape(np_array)[0]))
                print('   ---> {} dim'.format(np_array.ndim))
                print('           [{:15} / {:15} ]'.format('text', 'label'))
                print('           [{:15} / {:15} ]'.format(str(np_array[0][0].shape), str(np_array[0][1].shape)))
                print('           [{:15} / {:15} ]'.format(str(np_array[0][0].dtype), str(np_array[0][1].dtype)))

    if print_example:
        print('\n\n# Examples of data:')
        for i, ex in enumerate(np_array):
            print('{}'.format(pprint.pformat(ex)))
            if i + 1 > n_example:
                break

# print details on one example of the tokenize data
def print_detail_tokeniser(dataset, tokenizer, max_entries=20):
    np_array = np.array(list(dataset.take(1).as_numpy_iterator()))
    if len(np_array[0][0]['input_ids'].shape)==2:
        print('{:>10}     ---->    {:^15}   {:^15}   {:<30}\n'.format('input_ids', 'attention_mask', 'token_type_ids',
                                                                  'modified text'))
        for i, v in enumerate(np_array[0][0]['input_ids'][0]):
            print('{:>10}     ---->    {:^15d}   {:^15d}   {:<30}'.format(v,
                                                                          int(np_array[0][0]['attention_mask'][0][i]),
                                                                          int(np_array[0][0]['attention_mask'][0][i]),
                                                                          tokenizer.decode(int(v))))
            if i > max_entries:
                break
    elif len(np_array[0][0]['input_ids'].shape) == 1:
        print('{:>10}     ---->    {:^15}   {:^15}   {:<30}\n'.format('input_ids', 'attention_mask', 'token_type_ids',
                                                                        'modified text'))
        for i, v in enumerate(np_array[0][0]['input_ids']):
            print('{:>10}     ---->    {:^15d}   {:^15d}   {:<30}'.format(v,
                                                                          int(np_array[0][0]['attention_mask'][i]),
                                                                          int(np_array[0][0]['attention_mask'][i]),
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
def _gen(features):
    for f in features:
        yield ({"idx": f.idx,
                "label": f.label,
                "sentence": f.sentence,
                })

# transform and return data in the right structure
def _create_tf_example(idx, label, sentence):
    '''Puts the three inputs into the data structure required by GLUE and is called by the function convert_to_glue.'''

    features = []
    for i, x in enumerate(sentence):
        features.append(InputFeatures(np.int32(idx[i]), np.int64(label[i]), sentence[i]))

    return tf.data.Dataset.from_generator(
        lambda: _gen(features),
        ({"idx": tf.int32, "label": tf.int64, "sentence": tf.string}),
        ({"idx": tf.TensorShape([]),
          "label": tf.TensorShape([]),
          "sentence": tf.TensorShape([]),
          })
    )



def convert_np_array_to_glue_format(sentence, label, decode=False, shift=0):
    '''
    Description: This function converts a DatasetV1Adapter into a glue-compatible format by calculating 
                 the three main components from an input dataset. 
                 The function create_tf_example from preprocessing is called to create the final dataset.
    
    Args:
        sentence: numpy array either in byte utf-8 format or strings.
        label:    numpy array either in byte utf-8 format or integers.
        decode:   boolean which should be set to true if the numpy arrays that are passed are in byte format; 
                  default is false
        shift:    Parameter which ensures that indices for training and validation data set do not overlap.
                  At the moment, this parameter is used in the following way: 
                  1. convert the training dataset first without introducting a shift
                  2. convert the validation dataset with a shift of the length of the training dataset
    
    Outputs: FlatMapDataset which fits the following structure:
             {idx: (), label: (), sentence: ()}, types: {idx: tf.int32, label: tf.int64, sentence: tf.string}
    
    '''
    
    if decode:
        # get label
        to_int = lambda t: int(t.decode("utf-8"))
        label=list(map(to_int, label))

        # get idx
        idx=[j for j in range(0,len(label))]

        # get sentence
        to_string = lambda t: t.decode("utf-8")
        sentence=list(map(to_string, sentence))
    else:
        idx=[j+shift for j in range(0,len(label))]

    
    return _create_tf_example(idx, label, sentence)



def convert_tf_data_to_glue_format(data, shift=0):
    '''
    Description: This function converts a DatasetV1Adapter into a glue-compatible format by calculating 
                 the three main components from an input dataset. 
                 The function create_tf_example from preprocessing is called to create the final dataset.
    
    Args: 
        data:    DatasetV1Adapter, e.g. data['train'] when using an official Tensorflow dataset
        shift:   Parameter which ensures that indices for training and validation data set do not overlap.
                 At the moment, this parameter is used in the following way: 
                 1. convert the training dataset first without introducting a shift
                 2. convert the validation dataset with a shift of the length of the training dataset
    
    Outputs: FlatMapDataset which fits the following structure:
             {idx: (), label: (), sentence: ()}, types: {idx: tf.int32, label: tf.int64, sentence: tf.string}
    
    '''
    np_array=np.array(list(data.as_numpy_iterator()))
    
    # get label
    to_int = lambda t: int(t.decode("utf-8"))
    label=np_array[:,1].tolist()
    label=list(map(to_int, label))
    
    # get idx
    idx=[j+shift for j in range(0,len(label))]
    
    # get sentence
    to_string = lambda t: t.decode("utf-8")
    sentence=np_array[:,0].tolist()
    sentence=list(map(to_string, sentence))
    
    return _create_tf_example(idx, label, sentence)

def feature_selection(feature, label):
    print('feature:',feature['input_ids'],'label:',label)
    return feature['input_ids'], label


def label_extraction(feature, label):
    return label


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'input_ids': _bytes_feature(tf.io.serialize_tensor(features['input_ids'])),
        'attention_mask': _bytes_feature(tf.io.serialize_tensor(features['attention_mask'])),
        'token_type_ids': _bytes_feature(tf.io.serialize_tensor(features['token_type_ids'])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()

def generator(data):
    for features in data:
        yield serialize_example(*features)

def write_tf_data_into_tfrecord(data, file_name):
    serialized_features_dataset = tf.data.Dataset.from_generator(lambda: generator(data), output_types=tf.string, output_shapes=())

    filename = file_name + '.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

def parse_tfrecord_glue_files(record):
    # The tensors you pull into the model MUST have the same name
    # as what was encoded in the TFRecord

    # FixedLenFeature means that you know the number of tensors associated
    # with each label and example.

    # For example, there will only be 1 review per example, and as
    # a result, sentence is a FixedLenFeature.
    features_spec = {
        'input_ids': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'attention_mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'token_type_ids': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }

    # tr_parse_ds = tr_ds.map(parse_example)
    example = tf.io.parse_single_example(record, features_spec)

    f0 = tf.io.parse_tensor(example['input_ids'], out_type=tf.int32)
    f1 = tf.io.parse_tensor(example['attention_mask'], out_type=tf.int32)
    f2 = tf.io.parse_tensor(example['token_type_ids'], out_type=tf.int32)
    return {'input_ids': f0, 'attention_mask': f1, 'token_type_ids': f2}, example['label']