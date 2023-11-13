from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import joblib
from utils.helpers import get_res_df, get_standard_stats
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import os


def evaluate_lstm(database:str='mimic', 
                  seed:int=None, 
                  has_microbiology=True, 
                  inc_ab=False, 
                  use_censored = True,
                  lookback = 7, 
                  aggregated_hours = 4, 
                  num_lin_layers=1, 
                  num_stacked_lstm = 1,
                  hidden_dim = 128,
                  dropout_prob=0.3, 
                  lamb=1, 
                  is_tuned=False,
                  lr = 0.01,
                  bs = 64,
                  use_relus = False, 
                  use_batchnormalization = False):

    path = "data/results/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/use_censored_"+str(use_censored)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)+'/'+ \
            "dropout_"+str(dropout_prob).replace('.','-')+'/'+"lambda_"+str(lamb).replace('.','-')+'/'+"num_lin_layers_"+str(num_lin_layers)+'/' + \
            "num_stacked_lstm_"+str(num_stacked_lstm)+"/hidden_dim_"+str(hidden_dim)+"/lr_"+str(lr).replace('.','-')+"/bs_"+str(bs)+ \
            "/is_tuned_"+ str(is_tuned) + "/use_relus_"+ str(use_relus) + "/use_bn_"+ str(use_batchnormalization) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    
    # load test predictions
    test_res = pd.read_csv(path + 'test_gt_and_preds.csv')

    print("--- test eval ---")
    test_eval = get_standard_stats(gt=test_res['next_day'], preds=test_res['pred'], preds_proba=test_res['True'])
    test_eval.to_csv(path+"test_res.csv")

    print(test_eval)
    RocCurveDisplay.from_predictions(test_res['next_day'], test_res['True'])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
    plt.legend(loc="lower right")
    plt.savefig(path+'RocCurve.jpeg', format='jpeg')
    plt.close()

    PrecisionRecallDisplay.from_predictions(test_res['next_day'], test_res['True'])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    positive_class_frequency = sum(test_res['True']) / len(test_res['True'])
    plt.axhline(y=positive_class_frequency, color='r', linestyle='--', label='Random Classifier')
    plt.legend()
    plt.savefig(path+'PrecisionRecall.jpeg', format='jpeg')
    plt.close()
    



def run_evaluation(database='mimic', 
                   lookbacks=[2], 
                   prediction_time_points='random', 
                   numberofsamples=1, 
                   sample_train=None, 
                   sample_test=None, 
                   seed:int=None, 
                   inc_ab=True,
                   has_microbiology=False):

    from constant_files.traditional_models import get_models
    models = get_models(seed)
    
    if prediction_time_points == 'random':
        prediction_time_points = [('random', numberofsamples)]
    
    path = "data/results/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
    if not os.path.exists(path):
        os.makedirs(path)

    for lookback in lookbacks:
        for prediction_time_point in prediction_time_points:

            for key in models.keys():
                path_out = path + '/lookback_' + str(lookback) + '/time_point' + str(prediction_time_point) + '/sample_'+str(sample_train) +"_"+ str(sample_test)+ '/' +  str(key) + '/'

                # show the best parameters
                print("--- best params ---")
                print(pd.read_csv(path_out + 'best_params.csv'))

                # load train predictions
                train_res = pd.read_csv(path_out + 'train_gt_and_preds.csv')

                # load test predictions
                test_res = pd.read_csv(path_out + 'test_gt_and_preds.csv')

                train_eval = get_res_df(train_res)
                train_eval.to_csv(path_out+"train_res.csv")
                print("--- train eval ---")
                print(train_eval)

                print("--- test eval ---")
                test_eval = get_res_df(test_res)
                test_eval.to_csv(path_out+"test_res.csv")

                print(test_eval)
                RocCurveDisplay.from_predictions(test_res['lot<5d'], test_res['True'])
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=0.8)
                plt.legend(loc="lower right")
                plt.savefig(path_out+'RocCurve.jpeg', format='jpeg')
                plt.close()
                PrecisionRecallDisplay.from_predictions(test_res['lot<5d'], test_res['True'])
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                positive_class_frequency = sum(test_res['True']) / len(test_res['True'])
                plt.axhline(y=positive_class_frequency, color='r', linestyle='--', label='Random Classifier')
                plt.legend()
                plt.savefig(path_out+'PrecisionRecall.jpeg', format='jpeg')
                plt.close()

                # load the model
                model = joblib.load(path_out+'model.pkl')

                # load test set
                X_test = pd.read_parquet("data/model_input/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)+"/X_test_time_point_"+str(prediction_time_point).replace(".","-")+"_lookback_"+str(lookback)+".parquet")
                X_train = pd.read_parquet("data/model_input/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)+"/X_train_time_point_"+str(prediction_time_point).replace(".","-")+"_lookback_"+str(lookback)+".parquet")

                # if key == 'LogisticRegression':
                #     sample = shap.sample(X_train, 100)
                #     explainer = shap.KernelExplainer(model.predict_proba, sample)
                #     shap_values = explainer.shap_values(X_test)[1]
                #     shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='dot') #X_test
                #     plt.savefig(path_out+'shap_summary_plot_dot.jpeg', format='jpeg')
                #     plt.close()
                #     shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='violin')
                #     plt.savefig(path_out+'shap_summary_plot_violin.jpeg', format='jpeg')
                #     plt.close()
                #     pd.DataFrame(shap_values, columns=X_test.columns).to_csv(path_out+'shap_values.csv')
                # elif key == 'MLPClassifier':
                #     sample = shap.sample(X_train, 100)
                #     explainer = shap.KernelExplainer(model.predict_proba, sample)
                #     shap_values = explainer.shap_values(X_test)[1]
                #     shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='dot') #X_test
                #     plt.savefig(path_out+'shap_summary_plot_dot.jpeg', format='jpeg')
                #     plt.close()
                #     shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='violin')
                #     plt.savefig(path_out+'shap_summary_plot_violin.jpeg', format='jpeg')
                #     plt.close()
                #     pd.DataFrame(shap_values, columns=X_test.columns).to_csv(path_out+'shap_values.csv')
                if key == 'LGBMClassifier':
                    explainer = shap.Explainer(model, seed=seed)
                    shap_values = explainer.shap_values(X_test)[1]
                    shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='dot')
                    plt.savefig(path_out+'shap_summary_plot_dot.jpeg', format='jpeg')
                    plt.close()
                    shap_summary_plot = shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_type='violin')
                    plt.savefig(path_out+'shap_summary_plot_violin.jpeg', format='jpeg')
                    plt.close()
                    pd.DataFrame(shap_values, columns=X_test.columns).to_csv(path_out+'shap_values.csv')





# model should be e.g. LGBMClassifier
def run_test_loop(database='mimic', 
                  lookbacks=[2], 
                  prediction_time_points='random', 
                  numberofsamples=1, 
                  sample_train=None, 
                  sample_test=None, 
                  seed:int=None, 
                  inc_ab=True,
                  has_microbiology=False):
    
    from constant_files.traditional_models import get_models
    models = get_models(seed)
    
    if prediction_time_points == 'random':
        prediction_time_points = [('random', numberofsamples)]
    
    input_path = "data/model_input/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
    output_path = "data/results/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for lookback in lookbacks:
        for prediction_time_point in prediction_time_points:
            print("-------lookback: ",str(lookback),"--","time_point: "+str(prediction_time_point),"--", "sample_test: ",str(sample_test),"-------")

            common_str = "time_point_"+str(prediction_time_point).replace(".","-")+"_lookback_"+str(lookback)+".parquet"

            # load the prepared data
            X_train = pd.read_parquet(input_path+"/X_train_"+common_str)
            X_test = pd.read_parquet(input_path+"/X_test_"+common_str)
            y_train_raw = pd.read_parquet(input_path+"/y_train_"+common_str)
            y_test_raw = pd.read_parquet(input_path+"/y_test_"+common_str)


            # select our training target
            y_train = y_train_raw['lot<5d']
            y_test = y_test_raw['lot<5d']

            if sample_test != None:
                if sample_test == 'oversampling':
                    rs = RandomOverSampler(random_state=seed)
                    print("oversampler")
                elif sample_test == 'undersampling':
                    rs = RandomUnderSampler(random_state=seed)
                    print("undersampler")
                else:
                    raise Exception("sampler not supported")
                
                print("test set resampled")
                X_test, y_test = rs.fit_resample(pd.concat([X_test, y_test_raw], axis=1), y_test)
                y_test_raw = X_test[['lot<5d', 'lot_in_days', 'days_past']]
                X_test.drop(['lot<5d', 'lot_in_days', 'days_past'], axis=1,inplace=True)

            # for ever type of model
            for key in models.keys():
                print(key)

                # general path to all results
                path = output_path + '/lookback_' + str(lookback) + '/time_point' + str(prediction_time_point) + '/sample_'+str(sample_train) +"_"+ str(sample_test)+ '/' +  str(key) + '/'

                if not os.path.exists(path):
                    os.makedirs(path)

                # save the trained estimator
                model = joblib.load(path+'model.pkl')

                # calculate train set predictions
                pred_train = pd.DataFrame(model.predict(X_train), columns=['pred'])
                pred_proba_train = pd.DataFrame(model.predict_proba(X_train), columns=['False','True'])

                # save train set predictions
                train_gt_and_preds = pd.concat([y_train_raw.reset_index(drop=True), pred_train.reset_index(drop=True), pred_proba_train.reset_index(drop=True)], axis=1)
                train_gt_and_preds.to_csv(path+'train_gt_and_preds.csv', index=False)


                pred_test = pd.DataFrame(model.predict(X_test), columns=['pred'])
                pred_proba_test = pd.DataFrame(model.predict_proba(X_test), columns=['False','True'])
                
                test_gt_and_preds = pd.concat([y_test_raw.reset_index(drop=True), pred_test.reset_index(drop=True), pred_proba_test.reset_index(drop=True)], axis=1)
                test_gt_and_preds.to_csv(path+'/test_gt_and_preds.csv', index=False)









# ================== LSTM ====================
from train_module.lstm_trainer import create_dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import numpy as np
import torch
seed = 1
np.random.seed(seed)
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
#torch.cuda.set_device(0)  # if you have more than one CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_module.lstm_trainer import LSTMClassifier

from train_module.lstm_trainer import ID_COLS
from multiprocessing import cpu_count

def test_lstm(database:str='mimic', 
              fast=True, 
              seed:int=None, 
              has_microbiology=True, 
              use_censored=True,
              lookback = 7, 
              aggregated_hours = 4, 
              inc_ab=False, 
              num_lin_layers=1, 
              num_stacked_lstm = 1,
              hidden_dim = 128,
              dropout_prob=0.3, 
              lamb=1, 
              is_tuned=False,
              lr = 0.01,
              bs = 64,
              use_relus = False, 
              use_batchnormalization = False):

    path = "data/model_input/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)+"/"
    res_path = "data/results/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/use_censored_"+str(use_censored)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)+'/'+ \
               "dropout_"+str(dropout_prob).replace('.','-')+'/'+"lambda_"+str(lamb).replace('.','-')+'/'+"num_lin_layers_"+str(num_lin_layers)+'/' + \
               "num_stacked_lstm_"+str(num_stacked_lstm)+"/hidden_dim_"+str(hidden_dim)+"/lr_"+str(lr).replace('.','-')+"/bs_"+str(bs)+ \
               "/is_tuned_"+ str(is_tuned) + "/use_relus_"+ str(use_relus) + "/use_bn_"+ str(use_batchnormalization) + '/'

    print('Using device:', device)
    X_lstm_test = pd.read_parquet(path+"X_lstm_test.parquet")
    y_lstm_test = pd.read_parquet(path+"y_lstm_test.parquet")
    
    if use_censored == False:
        print("censored excluded")
        print("X before: ", X_lstm_test.shape)
        uncensored_series_ids = y_lstm_test[y_lstm_test['censored'] == False]['series_id'].unique()
        X_lstm_test = X_lstm_test[X_lstm_test['series_id'].isin(uncensored_series_ids)]
        y_lstm_test = y_lstm_test[y_lstm_test['censored'] == False]

    if fast:
        print("ATTENTION: fast mode acitvated")
        sample = list(y_lstm_test.sample(200, random_state=seed)['series_id'])
        X_lstm_test = X_lstm_test[X_lstm_test['series_id'].isin(sample)]
        y_lstm_test = y_lstm_test[y_lstm_test['series_id'].isin(sample)]
    else:
        print("ATTENTION: fast mode deacitvated")


    test_ds = create_dataset(X=X_lstm_test, y=y_lstm_test, dropcols=ID_COLS, seed=seed)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)


    model = LSTMClassifier(input_dim = X_lstm_test.shape[1]-3,  # number of features
                           hidden_dim = hidden_dim, 
                           num_stacked_lstm = num_stacked_lstm, 
                           num_lin_layers = num_lin_layers, 
                           dropout_prob = dropout_prob,
                           use_relus=use_relus,
                           use_batchnormalization=use_batchnormalization)
    model.load_state_dict(torch.load(res_path+'best.pth'))

    model = model.to(device)

    model.eval()

    correct, total = 0, 0
    predictions_list = []
    predictions_proba_list = []
    true_labels_list = []
    for x_test, y_test in test_dl:
        if is_tuned:
            temp = y_test[:, 2]
        else:
            temp = y_test[:, 0]
        x_test, y_test = [t.to(device) for t in (x_test, temp)]
        pred_next_day, pred_time_to_end = model(x_test.to(device))
        #pred_next_day = model(x_test)
        preds = F.log_softmax(pred_next_day, dim=1).argmax(dim=1)
        total += y_test.size(0)
        correct += (preds == y_test).sum().item()
        predictions_list.append(preds.cpu().numpy())
        true_labels_list.append(y_test.cpu().numpy())
        predictions_proba_list.append(F.softmax(pred_next_day, dim=1).cpu().detach().numpy()) #[:,1]

    acc = correct / total
    balanced_accuracy = balanced_accuracy_score(np.concatenate(true_labels_list), np.concatenate(predictions_list))
    print(f'Acc.: {acc:2.2%}. Bal. Acc.: {balanced_accuracy:2.2%}')


    pred_test = pd.DataFrame(np.concatenate(predictions_list), columns=['pred'])
    pred_proba_test = pd.DataFrame(np.concatenate(predictions_proba_list), columns=['False','True'])
    
    test_gt_and_preds = pd.concat([y_lstm_test.reset_index(drop=True), pred_test.reset_index(drop=True), pred_proba_test.reset_index(drop=True)], axis=1)
    test_gt_and_preds.to_csv(res_path+'/test_gt_and_preds.csv', index=False)






def compare_models(database='mimic', 
                  lookback=2, 
                  prediction_time_points='random', 
                  numberofsamples=1, 
                  sample_train=None, 
                  sample_test=None, 
                  seed:int=42, 
                  inc_ab=False,
                  has_microbiology=True,
                  model='LGBMClassifier',
                  dropout=0.3,
                  lamb=0.5,
                  num_relu_layers=2,
                  is_tuned=True,
                  lookback_lstm=7,
                  aggregated_hours=4):
    if prediction_time_points == 'random':
        time_point = ('random', numberofsamples)

    traditional_path = 'data/model_input/traditional/'+database+'/microbiology_res_'+str(has_microbiology)+'/ab_'+str(inc_ab)+'/seed_'+str(seed)+'/'
    traditional_model_path = 'data/results/traditional/'+database+'/microbiology_res_'+str(has_microbiology)+'/ab_'+str(inc_ab)+'/seed_'+str(seed)+ \
                            '/lookback_'+str(lookback)+'/time_point'+str(time_point)+'/sample_'+str(sample_train)+"_"+str(sample_test)+"/"+model+"/"
    
    result_path = 'data/results/combined/'+database+'/microbiology_res_'+str(has_microbiology)+'/ab_'+str(inc_ab)+'/seed_'+str(seed)+'/'+ \
                  '/lookback_'+str(lookback)+'/time_point'+str(time_point)+'/sample_'+str(sample_train)+'_'+str(sample_test)+'/'+model+'/' + \
                  '/dropout_'+str(dropout).replace('.','-')+'/lambda_'+str(lamb).replace('.','-')+'/num_relu_layers_'+str(num_relu_layers)+ \
                  '/is_tuned_'+str(is_tuned)+'/'
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #display(pd.read_parquet('data/model_input/traditional/'+database+'/microbiology_res_'+str(has_microbiology)+'/ab_'+str(inc_ab)+'/seed_'+str(seed)+'/'+'X_test_time_point_'+str(('random',1))+'_lookback_'+str(lookback)+'.parquet'))

    # first we get the complete test dataset for the traditional model
    X_traditional = pd.DataFrame()
    y_traditional = pd.DataFrame()
    for tp in [0,1,2,3,4]:
        X_part = pd.read_parquet(traditional_path+'X_test_time_point_'+str(tp)+'_lookback_'+str(lookback)+'.parquet')
        y_part = pd.read_parquet(traditional_path+'y_test_time_point_'+str(tp)+'_lookback_'+str(lookback)+'.parquet')

        X_traditional = pd.concat([X_traditional, X_part], axis=0, join='outer').fillna(0)
        y_traditional = pd.concat([y_traditional, y_part], axis=0, join='outer').fillna(0)

    #display(X_traditional)
    #display(y_traditional)

    # next we load the traditional model
    # save the trained estimator
    model = joblib.load(traditional_model_path+'model.pkl')

    # calculate test set predictions
    pred_test = pd.DataFrame(model.predict(X_traditional), columns=['pred'])
    pred_proba_test = pd.DataFrame(model.predict_proba(X_traditional), columns=['False','True'])

    #display(pred_test)
    #display(pred_proba_test)

    y_traditional = pd.concat([y_traditional.reset_index(drop=True), pred_test, pred_proba_test], axis=1)
    #display(y_traditional.groupby(['days_past']).count())

    #display(y_traditional.sort_values(['ID','days_past']))
    # get the predicitions from the next day model
    y_next_day = pd.read_csv('data/results/lstm/'+database+'/microbiology_res_'+str(has_microbiology)+'/ab_'+str(inc_ab)+"/lookback_"+str(lookback_lstm)+"/aggregated_hours_"+str(aggregated_hours)+'/seed_'+str(seed)+
                             '/dropout_'+str(dropout).replace('.','-')+'/lambda_'+str(lamb).replace('.','-')+'/num_relu_layers_'+str(num_relu_layers)+
                             '/is_tuned_'+str(is_tuned)+'/test_gt_and_preds.csv')
    
    y_next_day['starttime'] = pd.to_datetime(y_next_day['starttime'])
    y_next_day['starttime'] = y_next_day['starttime'].dt.floor('5min')
    #display(y_next_day.sort_values(['ID','days_past']))

    #display(y_next_day.groupby(['days_past']).count())

    combined = y_traditional.merge(y_next_day, on=['days_past', 'ID', 'starttime'], suffixes=('_traditional', '_lstm'))

    # pred_traditional True => stop therapy
    # pred_traditional False => complete therapy
    # pred_lstm True => complete therapy
    # pred_lstm False => stop therapy
    combined = combined[['ID','starttime','days_past','lot<5d','pred_traditional','pred_lstm','lot_in_days']]

    combined['pred_lstm'] = combined['pred_lstm'].astype(bool)
    combined['pred_lstm'] = ~combined['pred_lstm']
    # NOW:
    # True => stop therapy
    # False => complete therapy

    dict_df = {
        'days_past' : [],
        'median_num_prevented_unnecessary_ab_days_trad' : [],
        'median_num_prevented_unnecessary_ab_days_lstm': [],
        'median_num_missed_necessary_ab_days_trad': [],
        'median_num_missed_necessary_ab_days_lstm': [],
    }

    for days_past in range(0,5):
        t = combined[combined['days_past'] == days_past]

        dict_df['days_past'].append(days_past)

        #median_num_prevented_unnecessary_ab_days
        trad = t[(t['lot<5d'] == True) & (t['pred_traditional'] == True)].copy() # they got ab, did not needed them and traditional model said stop
        lstm = t[(t['lot<5d'] == True) & (t['pred_lstm'] == True)].copy() # they got ab, did not needed them and traditional model said stop

        trad['saved_days'] = trad['lot_in_days'] - days_past
        lstm['saved_days'] = lstm['lot_in_days'] - days_past

        dict_df['median_num_prevented_unnecessary_ab_days_trad'].append(trad['saved_days'].median())
        dict_df['median_num_prevented_unnecessary_ab_days_lstm'].append(lstm['saved_days'].median())

        #median_num_missed_necessary_ab_days
        trad = t[(t['lot<5d'] == False) & (t['pred_traditional'] == True)].copy()
        lstm = t[(t['lot<5d'] == False) & (t['pred_lstm'] == True)].copy()

        trad['missed_days'] = trad['lot_in_days'] - days_past
        lstm['missed_days'] = lstm['lot_in_days'] - days_past

        dict_df['median_num_missed_necessary_ab_days_trad'].append(trad['missed_days'].median())
        dict_df['median_num_missed_necessary_ab_days_lstm'].append(lstm['missed_days'].median())

    
    res = pd.DataFrame(dict_df)
    res.to_csv(result_path+'combined_result.csv')
    print(res)







def independent_test_nd(database='eicu', 
                  seed=44, 
                  model='LGBMClassifier',
                  is_tuned = True):


    independent_path = "data/model_input/lstm/"+database+"/microbiology_res_False/ab_False/lookback_7/aggregated_hours_4/seed_"+str(seed)+"/"

    
    
    
    # first we get the complete test dataset for the traditional model
    if database != 'mimic':
        X_independent = pd.read_parquet(independent_path+'X_lstm_train.parquet')
        y_independent = pd.read_parquet(independent_path+'y_lstm_train.parquet')

        # X_independent_test = pd.read_parquet(independent_path+'X_lstm_test.parquet')
        # y_independent_test = pd.read_parquet(independent_path+'y_lstm_test.parquet')

        # max_id = X_independent['series_id'].max()
        # X_independent_test['series_id'] = X_independent_test['series_id'] + max_id + 1
        # y_independent_test['series_id'] = y_independent_test['series_id'] + max_id + 1

        # X_independent = pd.concat([X_independent, X_independent_test], axis=0)
        # y_independent = pd.concat([y_independent, y_independent_test], axis=0)

    else:
        X_independent = pd.read_parquet(independent_path+'X_lstm_test.parquet')
        y_independent = pd.read_parquet(independent_path+'y_lstm_test.parquet')

    # print(X_independent.shape)
    # print(X_independent)
    # print(X_independent.columns)

    
    
    # print("------------")
    # print(X_independent.shape)
    # print(X_independent)
    # print(X_independent.columns)


    test_ds = create_dataset(X=X_independent, y=y_independent, dropcols=ID_COLS, seed=seed)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)


    model = LSTMClassifier(input_dim = X_independent.shape[1]-3,  # number of features
                           hidden_dim = 128, 
                           num_stacked_lstm = 1, 
                           num_lin_layers = 3, 
                           dropout_prob = 0.5,
                           use_relus=True,
                           use_batchnormalization=True)
    
    # load best mimic model
    #model_path = "data/results/lstm/mimic/microbiology_res_False/ab_False/use_censored_True/lookback_7/aggregated_hours_4/seed_"+str(seed)+"/dropout_0-0/lambda_0-1/num_lin_layers_2/num_stacked_lstm_3/hidden_dim_256/lr_0-01/bs_128/is_tuned_False/use_relus_False/use_bn_False/best.pth"
    model_path = "data/results/lstm/mimic/microbiology_res_False/ab_False/use_censored_True/lookback_7/aggregated_hours_4/seed_"+str(seed)+"/dropout_0-5/lambda_0-1/num_lin_layers_3/num_stacked_lstm_1/hidden_dim_128/lr_0-01/bs_128/is_tuned_False/use_relus_True/use_bn_True/best.pth"
    
    model.load_state_dict(torch.load(model_path)) #

    model = model.to(device)

    model.eval()

    correct, total = 0, 0
    predictions_list = []
    predictions_proba_list = []
    true_labels_list = []
    for x_test, y_test in test_dl:
        if is_tuned:
            temp = y_test[:, 2]
        else:
            temp = y_test[:, 0]
        x_test, y_test = [t.to(device) for t in (x_test, temp)]
        pred_next_day, pred_time_to_end = model(x_test.to(device))
        #pred_next_day = model(x_test)
        preds = F.log_softmax(pred_next_day, dim=1).argmax(dim=1)
        total += y_test.size(0)
        correct += (preds == y_test).sum().item()
        predictions_list.append(preds.cpu().numpy())
        true_labels_list.append(y_test.cpu().numpy())
        predictions_proba_list.append(F.softmax(pred_next_day, dim=1).cpu().detach().numpy())

    pred_test = pd.DataFrame(np.concatenate(predictions_list), columns=['pred'])
    pred_proba_test = pd.DataFrame(np.concatenate(predictions_proba_list), columns=['False','True'])
    
    #test_gt_and_preds = pd.concat([y_traditional_independent.reset_index(drop=True), pred_test.reset_index(drop=True), pred_proba_test.reset_index(drop=True)], axis=1)
    #test_gt_and_preds.to_csv(res_path+'/test_gt_and_preds.csv', index=False)
    #display(test_gt_and_preds)


    # calculate test set predictions
    #pred_test = pd.DataFrame(model.predict(X_traditional), columns=['pred'])
    #pred_proba_test = pd.DataFrame(model.predict_proba(X_traditional), columns=['False','True'])
    test_gt_and_preds = pd.concat([y_independent.reset_index(drop=True), pred_test.reset_index(drop=True), pred_proba_test.reset_index(drop=True)], axis=1)
    


    
    test_res = get_standard_stats(gt=test_gt_and_preds['next_day'], preds=test_gt_and_preds['pred'], preds_proba=test_gt_and_preds['True'])

    #display(test_res)
    test_gt_and_preds['seed'] = seed
    test_gt_and_preds['database'] = database
    test_res['seed'] = seed
    test_res['database'] = database
    return test_gt_and_preds, test_res


def independent():
    datasets = ['mimic', 'eicu', 'pic'] # 'pic' 'mimic'
    for dataset in datasets:
        print("dataset:", dataset)
        for seed in [42, 43, 44, 45, 46]: #, 43, 44, 45, 46
            print("seed:", seed)
            test_gt_and_preds, test_res = independent_test_nd(database=dataset, seed=seed)
            test_gt_and_preds.to_parquet("experiments/transferability/test_gt_and_preds_"+dataset+"_"+str(seed)+".parquet")
            test_res.to_parquet("experiments/transferability/test_res_"+dataset+"_"+str(seed)+".parquet")
