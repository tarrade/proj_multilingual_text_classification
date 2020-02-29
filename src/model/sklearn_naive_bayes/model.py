from joblib import parallel_backend
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import src.utils.ressources_utils as mem_utils
import src.analysis.get_data as get_data

def train_and_evaluate(eval_size, frac, max_df, min_df, norm, alpha, nb_label):
    
    # checking parameters dependencies
    #if min_df>max_df:
    #    print('---> min_df>=max_df: skipping')
    #    return None, 0.0
    #if (max_df-min_df)<0.1:
    #    print('---> (max_df-min_df)<0.1: skipping')
    #    return None, 0.0
    
    # transforming data type from YAML to python
    if norm=='None': norm=None     
    if min_df==1.0: min_df=1
    
    # print cpu info
    print('\n---> CPU ')
    mem_utils.info_cpu()
    
    # print mem info
    mem_utils.info_details_mem(text='---> details memory info: start')
    
    # print mem info
    mem_utils.info_mem(text=' ---> memory info: start') 
    
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
    
    use_pipeline = False
    
    if not use_pipeline:
        with parallel_backend('threading', n_jobs=16):
            print('joblib parallel_backend')
            try:
                # train
                cv = CountVectorizer(max_df=max_df,min_df=min_df,max_features=10000).fit(train_X)
                word_count_vector =cv.transform(train_X)
                print(' ---> Size CountVectorizer matrix')
                print('number of row {:,}'.format(word_count_vector.shape[0]))
                print('number of col {:,}'.format(word_count_vector.shape[1]))
                voc=cv.vocabulary_
                voc_list=sorted(voc.items(), key=lambda kv: kv[1], reverse=True)
                print(' --> length of the vocabulary vector: {:,}'.format(len(cv.get_feature_names())))
                #print(voc_list)

                # print mem info
                mem_utils.info_mem(text=' ---> memory info: after CountVectorizer')


                tfidf_transformer= TfidfTransformer(norm=norm).fit(word_count_vector)
                tfidf_vector=tfidf_transformer.transform(word_count_vector)
                print('cv tfidf', tfidf_vector.shape)
                #print(tfidf_vector)
                
                # print mem info
                mem_utils.info_mem(text=' ---> memory info: after TfidfTransformer')
            
                nb_model = MultinomialNB(alpha=alpha).fit(tfidf_vector, train_y)
                
                word_count_vector_eval =cv.transform(eval_X)
                tfidf_vector_eval=tfidf_transformer.transform(word_count_vector_eval)
            except:
                print('---> MultinomialNB(alpha=alpha).fit(tfidf_vector, train_y) is crashing ...')    
                return None, 0.0

    else:
        pipeline=Pipeline([('Word Embedding', CountVectorizer(max_df=max_df,min_df=min_df)),
                           ('Feature Transform', TfidfTransformer(norm=norm)),
                           ('Classifier', MultinomialNB(alpha=alpha))])
        
        try:
            pipeline.fit(train_X, train_y)
        except:
            print('---> pipeline.fit(train_X, train_y) is crashing ...')
            return None, 0.0

        print('the list of steps and parameters in the pipeline\n')
        for k, v in pipeline.named_steps.items():
            print('{}:{}\n'.format(k,v))

        # print the lenght of the vocabulary
        has_index=False
        if 'Word Embedding' in pipeline.named_steps.keys():
            # '.vocabulary_': dictionary item (word) and index 'world': index
            # '.get_feature_names()': list of word from (vocabulary)
            voc=pipeline.named_steps['Word Embedding'].vocabulary_
            voc_list=sorted(voc.items(), key=lambda kv: kv[1], reverse=True)
            print(' --> length of the vocabulary vector : \n{} {} \n'.format(len(voc), len(pipeline.named_steps['Word Embedding'].get_feature_names())))  

            # looking at the word occurency after CountVectorizer
            vect_fit=pipeline.named_steps['Word Embedding'].transform(eval_X)
            counts=np.asarray(vect_fit.sum(axis=0)).ravel().tolist()
            df_counts=pd.DataFrame({'term':pipeline.named_steps['Word Embedding'].get_feature_names(),'count':counts})
            df_counts.sort_values(by='count', ascending=False, inplace=True)
            print(' --> df head 20')
            print(df_counts.head(20))
            print(' --> df tail 20')
            print(df_counts.tail(20))
            print(' --- ')
            n=0
            for i in voc_list:
                n+=1
                print('    ',i)
                if (n>20):
                    break
            print(' --> more frequet words: \n{} \n'.format(voc_list[0:20]))
            print(' --- ')
            print(' --> less frequet words: \n{} \n'.format(voc_list[-20:-1]))
            print(' --- ')
            print(' --> longest word: \n{} \n'.format(max(voc, key=len)))
            print(' ---)')
            print(' --> shortest word: \n{} \n'.format(min(voc, key=len)))
            print(' --- ')
            index=pipeline.named_steps['Word Embedding'].get_feature_names()
            has_index=True

        # print the tfidf values
        if 'Feature Transform' in pipeline.named_steps.keys():
            tfidf_value=pipeline.named_steps['Feature Transform'].idf_
            #print('model\'s methods: {}\n'.format(dir(pipeline.named_steps['tfidf'])))
            if has_index:
                # looking at the word occurency after CountVectorizer
                tfidf_fit=pipeline.named_steps['Feature Transform'].transform(vect_fit)
                tfidf=np.asarray(tfidf_fit.mean(axis=0)).ravel().tolist()
                df_tfidf=pd.DataFrame({'term':pipeline.named_steps['Word Embedding'].get_feature_names(),'tfidf':tfidf})
                df_tfidf.sort_values(by='tfidf', ascending=False, inplace=True)
                print(' --> df head 20')
                print(df_tfidf.head(20))
                print(' --> df tail 20')
                print(df_tfidf.tail(20))
                print(' --- ')
                tfidf_series=pd.Series(data=tfidf_value,index=index)
                print(' --> IDF:')
                print(' --> Smallest idf:\n{}'.format(tfidf_series.nsmallest(20).index.values.tolist()))
                print(' {} \n'.format(tfidf_series.nsmallest(20).values.tolist()))
                print(' --- ')
                print(' --> Largest idf:\n{}'.format(tfidf_series.nlargest(20).index.values.tolist()))
                print('{} \n'.format(tfidf_series.nlargest(20).values.tolist()))
                print(' --- ') 
        
    # print mem info
    mem_utils.info_mem(text=' ---> memory info: after model training')

    # evaluate
    if not use_pipeline:
        train_y_pred = nb_model.predict(tfidf_vector)
    else:
        train_y_pred = pipeline.predict(train_X)
    
    # define the score we want to use to evaluate the classifier on
    acc_train = accuracy_score(train_y,train_y_pred)
    
    del train_X
    
    # evaluate
    if not use_pipeline:
        eval_y_pred = nb_model.predict(tfidf_vector_eval)
    else:
        eval_y_pred = pipeline.predict(eval_X)
    
    # define the score we want to use to evaluate the classifier on
    acc_eval = accuracy_score(eval_y,eval_y_pred)
    
    del eval_X
    
    # print mem info
    mem_utils.info_mem(text='---> memory info: after model evaluation')
    
    print('accuracy on test set: \n {} % \n'.format(acc_eval))
    print('accuracy on train set: \n {} % \n'.format(acc_train))
    if not use_pipeline:
        return nb_model, acc_eval
    else:
        return pipeline, acc_eval