import pandas as pd
import numpy as np
import joblib
import os

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import GridSearchCV

from constant_files.traditional_models import get_models


def run_train_loop(database='mimic', 
                   lookbacks=[2], 
                   prediction_time_points='random', 
                   numberofsamples=1, 
                   sample_train=None, 
                   sample_test=None, 
                   seed:int=None, 
                   inc_ab=True,
                   has_microbiology=False):
    
    models = get_models(seed)

    if prediction_time_points == 'random':
        prediction_time_points = [('random', numberofsamples)]

    input_path = "data/model_input/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
    output_path = "data/results/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for lookback in lookbacks:
        for prediction_time_point in prediction_time_points:
            print("-------lookback: ",str(lookback),"--","time_point: "+str(prediction_time_point),"--------")

            common_str = "time_point_"+str(prediction_time_point).replace(".","-")+"_lookback_"+str(lookback)+".parquet"

            # load the prepared data
            X_train = pd.read_parquet(input_path+"/X_train_"+common_str)
            X_test = pd.read_parquet(input_path+"/X_test_"+common_str)
            y_train_raw = pd.read_parquet(input_path+"/y_train_"+common_str)
            y_test_raw = pd.read_parquet(input_path+"/y_test_"+common_str)


            # select our training target
            y_train = y_train_raw['lot<5d']
            y_test = y_test_raw['lot<5d']
    
            
            # try setting is_unbalanced, try oversamping minority, try
            # FOR BALANCED EXPERIMENT 
            if sample_train != None:
                if sample_train == 'oversampling':
                    rs = RandomOverSampler(random_state=seed)
                    print("oversampler")
                elif sample_train == 'undersampling':
                    rs = RandomUnderSampler(random_state=seed)
                    print("undersampler")
                else:
                    raise Exception("sampler not supported")
                
                print("train set resampled")
                X_train, y_train = rs.fit_resample(X_train, y_train_raw['lot<5d'])
                #y_train = y_train_raw['lot<5d']
                #y_train_raw = X_train[['lot<5d', 'lot_in_days', 'days_past']]
                #X_train.drop(['lot<5d', 'lot_in_days', 'days_past'], axis=1,inplace=True)
                
            # for ever type of model
            for key in models.keys():
                print(key)
                # general path to all results
                path = output_path + '/lookback_' + str(lookback) + '/time_point' + str(prediction_time_point) + '/sample_'+str(sample_train) +"_"+ str(sample_test)+ '/' +  str(key) + '/'

                if not os.path.exists(path):
                    os.makedirs(path)

                # run a grid search
                gs = GridSearchCV(estimator=models[key]['model'], 
                                  param_grid=models[key]['params'], 
                                  scoring=['precision','recall','f1','balanced_accuracy','roc_auc','average_precision'], 
                                  cv=5, 
                                  refit='balanced_accuracy', 
                                  return_train_score=True)
                
                gs.fit(X_train, y_train)
                
                # save the trained estimator
                joblib.dump(gs.best_estimator_, path+'model.pkl')

                # grid search train and validation results
                train_valid_res = pd.DataFrame(gs.cv_results_)


                # save the grid search results
                train_valid_res.to_csv(path + 'train_valid_res.csv', index=False)


                # best estimator params
                #best_params = pd.DataFrame(gs.best_params_)
                print(gs.best_params_)
                best_params = pd.DataFrame(list(gs.best_params_.items()), columns=["Parameter", "Best Value"])

                best_params.to_csv(path+'best_params.csv', index=False)
                            