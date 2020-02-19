from collections import Counter
import operator
import google.cloud.bigquery as bigquery

def create_queries(eval_size):
    
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed`    
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query
    
def create_queries_subset(eval_size):
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query

def build_tag(row, list_tags):
    new_list=[]
    for idx, val in enumerate(row):
        if val in list_tags:
            new_list.append(val)
    del row
    return new_list

def query_to_dataframe(query, is_training, tags, nb_label):
    
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    
    #print(df['tags'])
    df['label'] = df['tags'].apply(lambda x: x[0] if len(x)>0 else 'other-tags')
    #print(df['label'])
    #df['label'] = df['tags'].apply(lambda row: ",".join(row))
    del df['tags']
    
    # features
    df['text'] = df['title'] + df['text_body'] + df['code_body']
    del df['code_body']
    del df['title']
    del df['text_body']
    
    # use BigQuery index
    df.set_index('id',inplace=True)
    
    keep_tags= ''
    
    return keep_tags, df


def create_dataframes(frac, eval_size, nb_label):   

    # small dataset for testing
    if frac > 0 and frac < 1:
        sample = " AND RAND() < {}".format(frac)
    else:
        sample = ""

    train_query, eval_query = create_queries(eval_size)
    train_query = "{} {}".format(train_query, sample)
    eval_query =  "{} {}".format(eval_query, sample)
    
    _, train_df = query_to_dataframe(train_query, True, '', nb_label)
    _, eval_df = query_to_dataframe(eval_query, False, '', nb_label)
    
    print('size of the training set          : {:,}'.format(len(train_df )))
    print('size of the evaluation set        : {:,}'.format(len(eval_df)))
    
    print('number of labels in training set  : {}'.format(len(train_df['label'].unique())))
    print('number of labels in evaluation set: {}'.format(len(eval_df['label'].unique())))
                                                              
    return train_df, eval_df


def input_fn(df):
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label