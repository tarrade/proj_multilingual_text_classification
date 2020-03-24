import tensorflow as tf
import numpy as np

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

