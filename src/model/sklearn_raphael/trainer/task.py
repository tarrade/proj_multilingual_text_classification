import argparse
import hypertune
import os

import trainer.model as model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_size',
        help = 'Size for the evaluation set in %',
        type = float,
        default=20.
    )
    parser.add_argument(
        '--frac',
        help = 'Fraction of input to process',
        type = float,
        default=0.0001
    )
    parser.add_argument(
        '--max_nb_label',
        help = 'Maximum number of labels',
        type = int,
        default=1000
    )
    parser.add_argument(
        '--WE_max_df',
        help='Document frequency strictly higher than the given threshold',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--WE_min_df',
        help='Document frequency strictly lower than the given threshold',
        type=float,
        default=1
    )        
    parser.add_argument(
        '--project_id',
        help='ID (not name) of your project',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='Output directory for model, automatically provided by gcloud',
        required=True
    )
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    #model.PROJECT = arguments['projectId']
    #model.KEYDIR  = 'trainer'
    
    print(arguments)
    
    estimator, acc_eval = model.train_and_evaluate(arguments['eval_size'],
                                                   arguments['frac'],
                                                   arguments['WE_max_df'],
                                                   arguments['WE_min_df'],
                                                   arguments['max_nb_label'])
    
    loc = model.save_model(estimator, 
                           arguments['job_dir'], 'stackoverlow')
    print("Saved model to {}".format(loc))
    
    # this is for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='balanced_accuracy',
        metric_value=acc_eval,
        global_step=0)