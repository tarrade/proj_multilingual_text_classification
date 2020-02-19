import os
import psutil
import subprocess
import datetime
import joblib
from collections import Counter
import operator
import copy
import pprint
import google.cloud.bigquery as bigquery
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)
import trainer.utils as utils

def create_queries(eval_size):
    
    query = """
    SELECT
      *
    FROM
      `nlp-text-classification.stackoverflow.posts_preprocessed_selection`    
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

    # split in df in training and testing
    
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
    #df = copy.deepcopy(input_df)
    
    # features, label
    label = df['label']
    del df['label']
    
    features = df['text']
    return features, label

def train_and_evaluate(eval_size, frac, max_df, min_df, nb_label):
    
    # print cpu info
    print('\n---> CPU ')
    utils.info_cpu()
    
    # print mem info
    utils.info_details_mem(text='---> details memory info: start')
    
   # print mem info
    utils.info_mem(text=' ---> memory info: start')
    
    # transforming data type from YAML to python
    if min_df==1.0: min_df=1
    
    # get data
    train_df, eval_df = create_dataframes(frac, eval_size, nb_label)
    utils.mem_df(train_df, text='\n---> memory training dataset')
    utils.mem_df(eval_df, text='\n---> memory evalution dataset')

    train_X, train_y = input_fn(train_df)
    eval_X, eval_y = input_fn(eval_df)
    
    del train_df
    del eval_df
    
    
    # print mem info
    utils.info_mem(text='\n---> memory info: after creation dataframe')
    
    # train
    estimators = [
    ('tfidf', TfidfVectorizer(tokenizer=lambda string: string.split(),
                              min_df=min_df, 
                              max_df=max_df,
                              ngram_range=(1,1))),
    ('clf', RandomForestClassifier(n_estimators=100,
                                   class_weight='balanced',
                                   n_jobs=-1))
    ]
    
    p = Pipeline(estimators)
    score = p.fit(train_X, train_y)
    eval_y_pred = p.predict(eval_X)
    
    # print mem info
    utils.info_mem(text=' ---> memory info: after model training')
    
    
    # define the score we want to use to evaluate the classifier on
    acc_eval = balanced_accuracy_score(eval_y,eval_y_pred)
    
    # print mem info
    utils.info_mem(text='---> memory info: after model evaluation')
    
    print('accuracy on test set: \n {} % \n'.format(acc_eval))
    print('accuracy on train set: \n {} % \n'.format(acc_train))

    return pipeline, acc_eval

def save_model(estimator, gcspath, name):
    
    model = 'model.joblib'
    joblib.dump(estimator, model)
    model_path = os.path.join(gcspath, datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    subprocess.check_call(['gsutil', '-o', 'GSUtil:parallel_composite_upload_threshold=150M', 'cp', model, model_path])
    return model_path