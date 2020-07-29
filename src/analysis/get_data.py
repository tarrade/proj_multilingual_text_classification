"""
Module contains helper functions to extract data.
Authors: Fabien Tarrade
"""
import google.cloud.bigquery as bigquery

def create_queries(eval_size):
    """
    Create queries to extract a evaluation and training dataset from BigQuery table.

    Parameters
    ----------
    eval_size : float
        fraction of the data to the evaluation dataset

    Returns
    -------
    eval_query: str
        query for the evaluation dataset
    train_query
        query for the training dataset
    """
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed`
    """

    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)

    return train_query, eval_query


def create_queries_subset(eval_size):
    """
    Create queries to extract a evaluation and training dataset from a preprocess BigQuery table.

    Parameters
    ----------
    eval_size : float
        fraction of the data to the evaluation dataset

    Returns
    -------
    eval_query: str
        query for the evaluation dataset
    train_query
        query for the training dataset
    """
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    """

    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)

    return train_query, eval_query


def build_tag(row, list_tags):
    """
    Create list of selected tags.

    Parameters
    ----------
    row : list
        list of tags
    list_tags: list
        tags to be used

    Returns
    -------
    new_list: list
        list of selected tags
    """
    new_list = []
    for idx, val in enumerate(row):
        if val in list_tags:
            new_list.append(val)
    del row
    return new_list


def query_to_dataframe(query):
    """
    Create queries to extract a evaluation and training dataset from a preprocess BigQuery table.

    Parameters
    ----------
    row : list
        list of tags
    list_tags: list
        tags to be used

    Returns
    -------
    new_list: list
        list of selected tags
    """
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    df['label'] = df['tags'].apply(lambda x: x[0] if len(x) > 0 else 'other-tags')
    del df['tags']

    # features
    df['text'] = df['title'] + df['text_body'] + df['code_body']
    del df['code_body']
    del df['title']
    del df['text_body']

    # use BigQuery index
    df.set_index('id', inplace=True)

    keep_tags = ''

    return keep_tags, df


def create_dataframes(frac, eval_size, nb_label):

    # small dataset for testing
    if frac > 0 and frac < 1:
        sample = " AND RAND() < {}".format(frac)
    else:
        sample = ""

    train_query, eval_query = create_queries(eval_size)
    train_query = "{} {}".format(train_query, sample)
    eval_query = "{} {}".format(eval_query, sample)

    _, train_df = query_to_dataframe(train_query, True, '', nb_label)
    _, eval_df = query_to_dataframe(eval_query, False, '', nb_label)

    print('size of the training set          : {:,}'.format(len(train_df)))
    print('size of the evaluation set        : {:,}'.format(len(eval_df)))

    print('number of labels in training set  : {}'.format(len(train_df['label'].unique())))
    print('number of labels in evaluation set: {}'.format(len(eval_df['label'].unique())))

    return train_df, eval_df


def input_fn(df):
    """
    Create features and labels dataframes.

    Parameters
    ----------
    df : pandas.DataFrame
        input pandas.DataFrame

    Returns
    -------
    features: pandas.DataFrame
        pandas.DataFrame with features
    label
        pandas.DataFrame with labels
    """
    # extract label
    label = df['label']
    del df['label']

    # extract features
    features = df['text']
    return features, label
