import os
import datetime
#import joblib
from sklearn.externals import joblib
import re
from google.cloud import storage

def save_model(estimator, gcspath, name):
    
    gcspath = re.sub('^gs:\/\/', '', gcspath)
    
    if len(gcspath.split('/'))<2:
        return 'ERROR: invalid path --> '+gcspath
    
    # Instantiates a client
    storage_client = storage.Client()

    # get the bucket
    bucket = storage_client.get_bucket(gcspath.split('/')[0])
    
    # extract the model
    model = 'model.joblib'
    joblib.dump(estimator, model)
    
    # save the model
    model_path = os.path.join('/'.join(gcspath.split('/')[1:]), datetime.datetime.now().strftime('export_%Y%m%d_%H%M%S'), model)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model)

    return 'gs://'+gcspath.split('/')[0]+model_path