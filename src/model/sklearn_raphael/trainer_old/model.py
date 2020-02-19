import os
import subprocess
import datetime
import joblib
import copy
import google.cloud.bigquery as bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

def create_queries(eval_size):
    #query = """
    #SELECT
    #*
    #FROM
    #`nlp-text-classification.stackoverflow.posts_preprocessed_selection_subset`
    #"""

    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed_selection`    
    """
    
    eval_query = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100) < {}".format(query, eval_size)
    train_query  = "{} WHERE MOD(ABS(FARM_FINGERPRINT(CAST(id as STRING))),100)>= {}".format(query, eval_size)
  
    return train_query, eval_query
    

def query_to_dataframe(query):
    
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    
    # label
    df['label'] = df['tags'].apply(lambda row: ",".join(row))
    del df['tags']
    
    # features
    df['text'] = df['title'] + df['text_body'] + df['code_body']
    del df['code_body']
    del df['title']
    del df['text_body']
    
    # use BigQuery index
    df.set_index('id',inplace=True)
    
    return df


def create_dataframes(frac, eval_size):   

    # split in df in training and testing
    #train_df, eval_df = train_test_split(df, test_size=0.2, random_state=101010)
    
    # small dataset for testing
    if frac > 0 and frac < 1:
        sample = " AND RAND() < {}".format(frac)
    else:
        sample = ""

    train_query, eval_query = create_queries(eval_size)
    train_query = "{} {}".format(train_query, sample)
    eval_query =  "{} {}".format(eval_query, sample)

    train_df = query_to_dataframe(train_query)
    eval_df = query_to_dataframe(eval_query)
    
    print('size of the training set  :',len(train_df ))
    print('size of the evaluation set:',len(eval_df ))
    print(train_df.info(memory_usage="deep"))
    return train_df, eval_df


def input_fn(input_df):
    df = copy.deepcopy(input_df)
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label

def train_and_evaluate(eval_size, frac, max_df, min_df, norm, alpha):
    
    # transforming data type from YAML to python
    if norm=='None': norm=None 
    if min_df==1.0: min_df=1
    
    # get data
    train_df, eval_df = create_dataframes(frac, eval_size)
    train_X, train_y = input_fn(train_df)
    
    # train
    pipeline=Pipeline([('Word Embedding', CountVectorizer(max_df=max_df,min_df=min_df)),
                       ('Feature Transform', TfidfTransformer(norm=norm)),
                       ('Classifier', MultinomialNB(alpha=alpha))])
    pipeline.fit(train_X, train_y)
   
    # evaluate
    eval_X, eval_y = input_fn(eval_df)
    eval_y_pred = pipeline.predict(eval_X)
    train_y_pred = pipeline.predict(train_X)

    # define the score we want to use to evaluate the classifier on
    acc_eval = accuracy_score(eval_y,eval_y_pred)

    print('accuracy on test set: \n {} % \n'.format(acc_eval))
    print('accuracy on train set: \n {} % \n'.format(accuracy_score(train_y,train_y_pred)))

    return pipeline, acc_eval

def save_model(estimator, gcspath, name):
    
    model = 'model.joblib'
    joblib.dump(estimator, model)
    model_path = os.path.join(gcspath, datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    subprocess.check_call(['gsutil', 'cp', model, model_path])
    return model_path