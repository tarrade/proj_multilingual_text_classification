import tensorflow as tf

def create_model(pretrained_weights, model_class, max_length, pretrained_model_dir, num_labels, learning_rate, epsilon, print_info=False):
    """Creates Keras Model for BERT Classification.
    Args:
      pretrained_weights
      pretrained_model_dir
      num_labels
      learning_rate,
      epsilon
    Returns:
      The compiled Keras model
    """

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # metric
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)

    # create model
    input_layer = tf.keras.Input(shape=(max_length,), dtype='int64')
    bert_ini = model_class.from_pretrained(pretrained_weights,
                                           cache_dir=pretrained_model_dir)(input_layer)
    # This is because in a bert pre-training progress, there are two tasks:
    # masked token prediction and next sentence prediction .
    # The first needs hidden state of each tokens ( shape: [batch_size, sequence_length, hidden_size])
    # the second needs the embedding of the whole sequence (shape : [batch_size, hidden_size] ) .
    bert = bert_ini[1]
    dropout = tf.keras.layers.Dropout(0.1)(bert)
    flat = tf.keras.layers.Flatten()(dropout)
    classifier = tf.keras.layers.Dense(units=num_labels)(flat)  # activation='softmax'

    if print_info:
        print('bert_ini[0]:',bert_ini[0])
        print('bert_ini[1]:', bert_ini[1])

    model = tf.keras.Model(inputs=input_layer,
                           outputs=classifier,
                           name='custom_tf_bert_classification')

    # compile Keras model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])

    return model