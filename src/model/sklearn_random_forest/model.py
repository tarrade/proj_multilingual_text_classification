from joblib import parallel_backend
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

import utils.memory_utils as mem_utils
#import utils.model_utils as model_utils
import analysis.get_data as get_data

def train_and_evaluate(eval_size, frac, max_df, min_df, norm, ngram_range, nb_label):
    
    # print cpu info
    print('\n---> CPU ')
    mem_utils.info_cpu()
    
    # print mem info
    mem_utils.info_details_mem(text='---> details memory info: start')
    
    # print mem info
    mem_utils.info_mem(text=' ---> memory info: start')
    
    # transforming data type from YAML to python
    if norm=='None': norm=None 
    if min_df==1.0: min_df=1
    
    # get data
    train_df, eval_df = get_data.create_dataframes(frac, eval_size, nb_label)
    mem_utils.mem_df(train_df, text='\n---> memory training dataset')
    mem_utils.mem_df(eval_df, text='\n---> memory evalution dataset')

    train_X, train_y = get_data.input_fn(train_df)
    eval_X, eval_y = get_data.input_fn(eval_df)
    
    del train_df
    del eval_df
    
    # print mem info
    mem_utils.info_mem(text='\n---> memory info: after creation dataframe')
    
    
    estimators = [
    ('tfidf-vectorizer', TfidfVectorizer(tokenizer=lambda string: string.split(),
                                         min_df=min_df, 
                                         max_df=max_df,
                                         ngram_range=ngram_range,
                                         norm=norm)),
        
    ('random-forest-classifier', RandomForestClassifier(n_estimators=100,
                                                        class_weight='balanced',
                                                        n_jobs=-1))
    ]
    
    p = Pipeline(estimators)
    
    try:
        p.fit(train_X, train_y)
    except:
        print('---> pipeline.fit(train_X, train_y) is crashing ...')
        return None, 0.0
    
    eval_y_pred = p.predict(eval_X)
    train_y_pred = p.predict(train_X)
    
    #loc = model_utils.save_model(p, 
    #                             arguments['job_dir'], 'stackoverlow')
    #print("Saved model to {}".format(loc))
    
    # print mem info
    mem_utils.info_mem(text=' ---> memory info: after model training')
    
    
    # define the score we want to use to evaluate the classifier on
    acc_eval = balanced_accuracy_score(eval_y,eval_y_pred)
    acc_train = balanced_accuracy_score(train_y, train_y_pred)
    
    # print mem info
    mem_utils.info_mem(text='---> memory info: after model evaluation')
    
    print('accuracy on test set: \n {} % \n'.format(acc_eval))
    print('accuracy on train set: \n {} % \n'.format(acc_train))

    return p, acc_eval