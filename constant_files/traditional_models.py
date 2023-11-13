#import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

def get_models(seed:int):

    models = {
                'LGBMClassifier' : {
                    'model' : lgb.LGBMClassifier(),
                    'params' : {
                        'reg_alpha': [0, 0.01, 0.1, 1, 10, 100], #, 1000
                        'reg_lambda': [0, 0.01, 0.1, 1, 10, 100], #, 1000
                        'random_state': [seed]
                    },
                    'best' : None,
                    'gs_valid_res_df': None,
                    'train_res_df' : None,
                    'test_res_df' : None
                },
                'LogisticRegression' : {
                    'model' : LogisticRegression(),
                    'params' : {
                        'penalty' : ['l2'],
                        'C': [0.001, 0.01, 0.1],  #0.0001, , 1, 10, 100, 1000
                        'random_state': [seed],
                        'solver' : ['liblinear']
                    },
                    'best' : None,
                     'gs_valid_res_df': None,
                     'train_res_df' : None,
                     'test_res_df' : None
                },
                'MLPClassifier' : {
                    'model' : MLPClassifier(),
                    'params' : {
                        'alpha': [0.0001, 0.001, 0.01], 
                        'hidden_layer_sizes': [(100), (100,10), (100, 50, 10)],
                        'random_state': [seed]
                    },
                    'best' : None,
                    'gs_valid_res_df': None,
                    'train_res_df' : None,
                    'test_res_df' : None
                }
            }
    
    return models