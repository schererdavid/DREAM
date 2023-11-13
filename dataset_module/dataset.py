import pandas as pd
import numpy as np
import json
import random
from sklearn.impute import SimpleImputer
import joblib

import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def append_dataframe_multiple_times(df, num_times):
    if num_times <= 0:
        raise ValueError("num_times must be greater than zero")

    new_df = pd.concat([df] * num_times, ignore_index=True)
    return new_df


def generate_random_day_past(row):
    return pd.Timedelta(days=min(row['lot_in_days']-1, 5-1) * random.random()) # -1 corrects for the minimum treatment duration of one day

def construct_X_y(database:str, time_window_length:int, days_past_first_application:any, episodes:pd.DataFrame, inc_ab:bool, filling='zero'):
    # because we mapped all features to a 5min slots we do this also here
    cols = ['AdmissionDateTime_hosp','endtime','starttime','lot','DischargeDateTime_hosp','AdmissionDateTime_icu','DischargeDateTime_icu']
    episodes[cols] = episodes[cols].apply(lambda x: x.dt.floor('5min')) 


    if type(days_past_first_application) not in [int, float]:
        # it's a tuple of the form ('random','int')
        episodes = append_dataframe_multiple_times(episodes, days_past_first_application[1])
        y = episodes[['lot<5d','lot_in_days']].copy()
        ep = episodes[['ID','starttime','lot_in_days']].copy()
        y['days_past'] = ep.apply(generate_random_day_past, axis=1)
    else:
        # it's the normal integer version
        episodes = episodes[episodes['lot_in_days'] >= days_past_first_application]
        y = episodes[['lot<5d','lot_in_days']].copy()
        ep = episodes[['ID','starttime','lot_in_days']].copy()
        y['days_past'] = pd.Timedelta(days_past_first_application, 'days')

    y['days_past'] = y['days_past'].apply(lambda x: x.total_seconds() / (3600 * 24))
    ep['X_end'] = ep['starttime'] + pd.to_timedelta(y['days_past'], unit='D')
    ep['X_start'] = ep['X_end'] - pd.Timedelta(time_window_length, 'days')
    y[['ID','starttime']] = ep[['ID','starttime']]
    ep.drop(['starttime','lot_in_days'], axis=1, inplace=True)

    X = ep.copy()


    with open(database+'_feature_list.json', 'r') as file:
        features = json.load(file)

    if not inc_ab:
        print("Antibiotics excluded as features")   
        del features['AB_applied.parquet']
    else:
        print("Antibiotics included as features")   

    for feature in features.keys():
        feature_name = feature.split('.')[0]

        if features[feature]['type'] == 'constant':
            feature_df = pd.read_parquet('data/features/'+database+'/'+feature)
            # filter patients

            X = pd.merge(X, feature_df, how='left', on='ID')
        else:
            feature_df = pd.read_parquet('data/features/'+database+'_5min/'+feature)
            # filter patients
            pat_rest = pd.merge(ep, feature_df, how='inner', on='ID')
            # filter time window
            tw_rest = pat_rest[(pat_rest['X_start'] <= pat_rest['Time']) & (pat_rest['Time'] <= pat_rest['X_end'])].copy().drop_duplicates()

            if tw_rest.shape[0] == 0:
                continue
            
            tw_rest['relative_time'] = tw_rest['X_end'] - tw_rest['Time']
            tw_rest.drop(['Time'], axis=1, inplace=True)
            # aggregate hours
            aggregated_hours = features[feature]['aggregated_hours']

            if aggregated_hours != None:
                # we bin the time and aggregate
                bin_edges = range(0, int((time_window_length*24*60*60) / 3600) + aggregated_hours + 1, aggregated_hours)
                bin_labels = range(len(bin_edges) - 1)
                tw_rest['relative_time_binned'] =  pd.cut(tw_rest['relative_time'].dt.total_seconds() / 3600, bins=bin_edges, labels=bin_labels, right=False) 
                tw_rest.drop(['relative_time'], axis=1, inplace=True)
                tw_rest['relative_time_binned'] = tw_rest['relative_time_binned'].astype(int)
                tw_rest = tw_rest.groupby(['ID', 'X_start', 'X_end', 'relative_time_binned']).agg({
                    tw_rest.columns[3]: 'mean',
                    tw_rest.columns[4]: 'sum'
                }).reset_index()
                # then pivot
                pivoted = tw_rest.pivot(index=['ID', 'X_start', 'X_end'], columns=["relative_time_binned"], values=tw_rest.columns[-2:])
                L = pd.cut(range(0, (int(time_window_length*24*60*60 / 3600) + 1), aggregated_hours), bins=bin_edges, labels=bin_labels, right=False).astype(int)

                existing_columns = pivoted.columns.get_level_values(1).unique()

                missing_columns = [col for col in L if col not in existing_columns]
                # add the missing columns
                new_columns = pd.MultiIndex.from_product([pivoted.columns.levels[0], missing_columns], names=pivoted.columns.names)
                pivoted = pivoted.reindex(columns=pivoted.columns.union(new_columns))
                pivoted = pivoted.sort_index(axis=1, ascending=True)

                # fill NAs
                count_cols = pivoted.columns.get_level_values(0).str.endswith('_count')
                pivoted.loc[:, count_cols] = pivoted.loc[:, count_cols].fillna(0)
                mean_cols = pivoted.columns.get_level_values(0).str.endswith('_mean')
                pivoted.loc[:, mean_cols] = pivoted.loc[:, mean_cols].ffill(axis=1)
                pivoted.loc[:, mean_cols] = pivoted.loc[:, mean_cols].bfill(axis=1)
                pivoted = pivoted.reset_index()
                pivoted.columns = pivoted.columns.get_level_values(0) + '_' + pivoted.columns.get_level_values(1).astype(str)
                pivoted.rename(columns={'ID_':'ID','X_start_':'X_start','X_end_':'X_end'}, inplace=True)
                X = pd.merge(X, pivoted, how='left', on=['ID','X_start','X_end'])

            else:
                bin_edges = range(0, time_window_length*24*12 + 5, 5)
                bin_labels = range(len(bin_edges) - 1)
                tw_rest = tw_rest.sort_values(['relative_time'])
                tw_rest['relative_time_binned'] =  pd.cut(tw_rest['relative_time'].dt.total_seconds() / (60*5), bins=bin_edges, labels=bin_labels, right=False)
                tw_rest.drop(['relative_time'], axis=1, inplace=True)
                tw_rest['relative_time_binned'] = tw_rest['relative_time_binned'].astype(int)
                tw_rest = tw_rest.groupby(['ID', 'X_start', 'X_end', 'relative_time_binned']).agg({
                    tw_rest.columns[3]: 'mean',
                    tw_rest.columns[4]: 'sum'
                }).reset_index() 
                # then  pivot
                pivoted = tw_rest.pivot(index=['ID', 'X_start', 'X_end'], columns=["relative_time_binned"], values=tw_rest.columns[-2:])
                L = pd.cut(range(0, time_window_length*24*12 + 1, 5), bins=bin_edges, labels=bin_labels, right=False).astype(int)

                existing_columns = pivoted.columns.get_level_values(1).unique()

                missing_columns = [col for col in L if col not in existing_columns]
                # add the missing columns
                new_columns = pd.MultiIndex.from_product([pivoted.columns.levels[0], missing_columns], names=pivoted.columns.names)
                pivoted = pivoted.reindex(columns=pivoted.columns.union(new_columns))
                pivoted = pivoted.sort_index(axis=1, ascending=True)
                # fill NAs
                count_cols = pivoted.columns.get_level_values(0).str.endswith('_count')
                pivoted.loc[:, count_cols] = pivoted.loc[:, count_cols].fillna(0)
                mean_cols = pivoted.columns.get_level_values(0).str.endswith('_mean')

                if filling == 'zero':
                    pivoted.loc[:, mean_cols] = pivoted.loc[:, mean_cols].fillna(0)
                elif filling == 'method_1':
                    pivoted.loc[:, mean_cols] = pivoted.loc[:, mean_cols].ffill(axis=1)
                    pivoted.loc[:, mean_cols] = pivoted.loc[:, mean_cols].bfill(axis=1)    
                else:
                    raise Exception("filling method not supported")
                          
                pivoted = pivoted.reset_index()
                pivoted.columns = pivoted.columns.get_level_values(0) + '_' + pivoted.columns.get_level_values(1).astype(str)
                pivoted.rename(columns={'ID_':'ID','X_start_':'X_start','X_end_':'X_end'}, inplace=True)     
                X = pd.merge(X, pivoted, how='left', on=['ID','X_start','X_end'])


    return X.reset_index(drop=True), y.reset_index(drop=True)



def check_output_for_0(window_length_list, days_past_first_application_list, path_to_X, percentage):
    for window_length in window_length_list:
        for days_past_first_application in days_past_first_application_list: 
            common_str = "days_past_"+str(days_past_first_application).replace(".","-")+"_window_length_"+str(window_length).replace(".","-")+".parquet"
            X_train_filled = pd.read_parquet(path_to_X+"/X_"+common_str)
            count_cols = X_train_filled.columns.get_level_values(0).str.contains("_count_")
            counts = X_train_filled.loc[:, count_cols]

            zeros = pd.DataFrame((counts == 0).sum(), columns=['count']).sort_values(['count'], ascending=True)
            zeros['percentage'] = zeros['count'] / X_train_filled.shape[0]
            
            print("window_length",window_length,"---","days_past_first_application",days_past_first_application)        
            print(zeros.shape, zeros[zeros['percentage'] > percentage].shape)


def print_dataset_statistics(X, y, name):
    print("---",name,"---\n")
    print("X.shape:", X.shape, "\n")
    print("#0:", np.count_nonzero(y == 0), "\n")
    print("#1:", np.count_nonzero(y == 1), "\n")
    print("#0/#total:", np.count_nonzero(y == 0)/(np.count_nonzero(y == 0)+np.count_nonzero(y == 1)), "\n")
    print("#1/#total:", np.count_nonzero(y == 1)/(np.count_nonzero(y == 0)+np.count_nonzero(y == 1)), "\n")


from utils.helpers import remove_special_characters

def construct_input_traditional(database='mimic', 
                                lookbacks=[2], 
                                prediction_time_points='random', 
                                numberofsamples=1, 
                                seed=None, 
                                inc_ab=True,
                                has_microbiology=False,
                                filling = 'zero'):
    path = 'data/episodes/'+database+'/microbiology_res_'+str(has_microbiology)+'/seed_'+str(seed)
    
    if prediction_time_points == 'random':
        prediction_time_points = [('random', numberofsamples)]


    df_train = pd.read_parquet(path+'/train_data.parquet')
    df_test = pd.read_parquet(path+'/test_data.parquet')

    # explanation: when the lot is larger than 5 days we do not care anymore if the patient was censored or not: lot<5d will be False
    # when lot is smaller than 5 days we can only label the uncensored ones
    # so exclude less than 5 days lot & censored
    df_train = df_train[(~df_train['censored']) | (df_train['lot_in_days'] >= 5)]
    df_test = df_test[(~df_test['censored']) | (df_test['lot_in_days'] >= 5)]
    df_train['lot<5d'] = df_train['lot_in_days'] < 5
    df_test['lot<5d'] = df_test['lot_in_days'] < 5

    print("df_train", df_train.shape)
    print("df_test", df_test.shape)

    

    for lookback in lookbacks: # [2]
        for prediction_time_point in prediction_time_points: # [0,1,2,3,4]: #[0, 1, 1.5, 2, 2.5, 3]: 0.5,1,2,3,4, ('random',1), ('random',3)
            print(prediction_time_point)

            if filling == 'zero':
                imputer = SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)
            elif filling == 'method_1':
                imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        
            
            X_train, y_train = construct_X_y(database=database, 
                                             time_window_length=lookback, 
                                             days_past_first_application=prediction_time_point, 
                                             episodes=df_train, 
                                             inc_ab=inc_ab)
            #y_train[['ID','X_end','X_start']] = X_train[['ID','X_end','X_start']]

            X_train.drop(['ID','X_end','X_start'], axis=1, inplace=True)
            X_train_filled = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)



            # -------- test -----------

            #df_test_after_hours_passed = df_test[df_test['lot_in_days'] >= days_past_first_application]
            #y_test = df_test_after_hours_passed[['lot<5d','lot_in_days']]
            if type(prediction_time_point) not in [int, float]:
                X_test, y_test = construct_X_y(database=database, 
                                               time_window_length=lookback, 
                                               days_past_first_application=(prediction_time_point[0], 1), 
                                               episodes=df_test, 
                                               inc_ab=inc_ab)
            else:
                X_test, y_test = construct_X_y(database=database, 
                                               time_window_length=lookback, 
                                               days_past_first_application=prediction_time_point, 
                                               episodes=df_test, 
                                               inc_ab=inc_ab)
            #y_test[['ID','X_end','X_start']] = X_test[['ID','X_end','X_start']]
            X_test = X_test.drop(['ID','X_end','X_start'], axis=1).copy()
            

            missing_columns = set(X_train.columns) - set(X_test.columns)
            extra_columns = set(X_test.columns) - set(X_train.columns)


            for column in missing_columns:
                X_test[column] = np.nan


            X_test = X_test.drop(columns=extra_columns)

            # sort the columns
            X_test = X_test[X_train.columns]

            # impute missing values
            X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

            # ----- save constructed data ----
            common_str = "time_point_"+str(prediction_time_point).replace(".","-")+"_lookback_"+str(lookback).replace(".","-")+".parquet"



            # clean up the column names
            X_train_filled.columns = [remove_special_characters(col) for col in X_train_filled.columns]
            X_test_filled.columns = [remove_special_characters(col) for col in X_test_filled.columns]
            # scale X
            scaler = StandardScaler()
            inf_columns = X_train_filled.columns[(X_train_filled.applymap(np.isinf)).any()].tolist()
            X_train_filled = pd.DataFrame(scaler.fit_transform(X_train_filled), columns=X_train_filled.columns)
            X_test_filled = pd.DataFrame(scaler.transform(X_test_filled), columns=X_test_filled.columns) 

            dest_path = "data/model_input/traditional/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/seed_"+str(seed)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)

            # save train data
            X_train_filled.to_parquet(dest_path+"/X_train_"+common_str)
            y_train.to_parquet(dest_path+"/y_train_"+common_str)

            # save test data
            X_test_filled.to_parquet(dest_path+"/X_test_"+common_str)
            y_test.to_parquet(dest_path+"/y_test_"+common_str)

            print("X_train", X_train_filled.shape, "| y_train", y_train.shape, "| X_test", X_test_filled.shape, "| y_test", y_test.shape, )





import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
seed = 1
np.random.seed(seed)
#torch.cuda.set_device(0)  # if you have more than one CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler





def convert_to_next_day(path:str, censor_column="DischargeDateTime_hosp"):
    ep_meta = pd.read_parquet(path)

    # censored if df['endtime'] + pd.Timedelta(2, 'days') > df['DischargeDateTime_hosp']
    # => we cannot be sure that the episode really ended at endtime

    # from ep_meta construct all construct the labels
    # for x in {0,1,2,3,4}
    # e.g. for t_x set label to True if (starttime + x*24h + 24h <= endtime),
    #              set label to False if (starttime + x*24h + 24h > endtime) and (starttime + x*24h + 24h <= DischargeDateTime_hosp)
    #              set label to NaN otherwise (it is censored)

    df = pd.DataFrame()

    for i in range(0,5):
        # first we only look for the next day label
        # it is clearly true if the endtime was larger than tomorrow
        now = ep_meta['starttime'] + i * pd.Timedelta(1, "days")
        trues = ep_meta[now + pd.Timedelta(1, "days") <= ep_meta['endtime']].copy()
        # it is clearly false if the endtime was smaller than tomorrow and the patient was not disscharged at this time
        falses = ep_meta[(now + pd.Timedelta(1, "days") > ep_meta['endtime']) & 
                         (now + pd.Timedelta(1, "days") <= ep_meta[censor_column])].copy()

        true_ids = trues.index.tolist()
        false_ids = falses.index.tolist()

        filtered_ep_meta = ep_meta[~ep_meta.index.isin(true_ids + false_ids)]
        unkowns = filtered_ep_meta.copy()

        unkowns = unkowns.reset_index(drop=True)

        trues['next_day'] = False
        falses['next_day'] = True
        unkowns['next_day'] = pd.NA

        temp = pd.concat([trues, falses, unkowns])
        temp['days_past'] = i

        # add the rest length of treatment in days
        # lot_in_days = end - start + 1
        temp['rest_lot_in_days'] = temp['lot_in_days'] - i

        temp['next_day_tuned'] = temp['lot_in_days'] < 5

        temp['rest_lot_in_days_tuned'] = temp.apply(lambda row: -i if row['next_day_tuned'] else row['rest_lot_in_days'], axis=1)

        df = pd.concat([df, temp])

    df['pred_time'] = df['starttime'] + df['days_past'] * pd.Timedelta(1, "days")

    # we drop the entries where we do not even know the next day label

    df = df[df['next_day'].notna()]


    # in reality it makes no sense to predict the next day when the patient already had a prediction to stop
    # we would also train with data where the patient did not got AB anymore
    # all the Falses
    next_day_false_rows = df[df['next_day'] == False]
    # the first True/Stop prediction of the series if there is any
    min_days_past_rows = df[df['next_day'] == True].groupby(['ID', 'starttime']).apply(lambda x: x[x['days_past'] == x['days_past'].min()]).reset_index(drop=True)
    df = pd.concat([next_day_false_rows, min_days_past_rows]).sort_values(by=['ID', 'days_past'])

    

    df = df[['ID','next_day','days_past','pred_time', 'starttime', 'censored','rest_lot_in_days','next_day_tuned','rest_lot_in_days_tuned']]

    df = df.reset_index(drop=True).reset_index().rename({'index':'series_id'}, axis=1)
    
    return df





'''
24/number_of_aggregated_hours must equal an integer
'''
def construct_X_lstm_helper(database, df:pd.DataFrame, lookback_days=7, number_of_aggregated_hours=4, inc_ab=True):

    max_measurements = int((lookback_days*24)/number_of_aggregated_hours)
    print("max number of measurements: ", max_measurements)

    measurement_data = []
    for series_id in df['series_id']:
        measurement_data.extend([(series_id, measurement_number) for measurement_number in range(max_measurements)])

    X = pd.DataFrame(measurement_data, columns=['series_id', 'measurement_number'])

    df = df[['series_id','ID','pred_time']].copy()
    with open(database+'_feature_list.json', 'r') as file:
        features = json.load(file)

    if not inc_ab:
        print("Antibiotics excluded as features")   
        del features['AB_applied.parquet']
    else:
        print("Antibiotics included as features") 

    for feature in tqdm(features, desc="Processing features"):
        #print(feature)
        #feature = 'temperature.parquet'

        if features[feature]['type'] == 'numerical':
            feature_df = pd.read_parquet('data/features/'+database+'_5min/'+feature)

            # drop count column
            # feature_df = feature_df[feature_df.columns[:-1]]

            # filter patients
            pat_rest = pd.merge(df, feature_df, how='inner', on='ID')
            pat_rest.drop(['ID'], axis=1, inplace=True)


            # filter lookback
            tw_rest = pat_rest[(pat_rest['pred_time'] - pd.Timedelta(lookback_days,'days') <= pat_rest['Time']) & (pat_rest['Time'] <= pat_rest['pred_time'])].copy()

            tw_rest['relative_time_to_pred'] = tw_rest['pred_time'] - tw_rest['Time']
            tw_rest['relative_time_since_lookback'] = pd.Timedelta(lookback_days, 'days') - tw_rest['relative_time_to_pred']
            tw_rest.drop(['Time'], axis=1, inplace=True)

            #number_of_bins = lookback_days * (24 / number_of_aggregated_hours)
            tw_rest['number_of_hours'] = tw_rest['relative_time_since_lookback'].dt.total_seconds() / 3600

            tw_rest['measurement_number'] = ((tw_rest['number_of_hours'] / number_of_aggregated_hours)).apply(np.floor).astype(int)

            tw_rest.drop(['relative_time_to_pred','relative_time_since_lookback','number_of_hours','pred_time'], axis=1, inplace=True)

            # group them
            tw_rest = tw_rest.groupby(['series_id','measurement_number']).mean().reset_index()

            #print(tw_rest)
            X = X.merge(tw_rest, on=["series_id", "measurement_number"], how="left")
            #print(X)

        else:
            feature_df = pd.read_parquet('data/features/'+database+'/'+feature)
            pat_rest = pd.merge(df, feature_df, how='inner', on='ID')
            pat_rest.drop(['pred_time','ID'], axis=1, inplace=True)
            pat_rest.iloc[:, 1] = pat_rest.iloc[:, 1].astype(float)
            #print(pat_rest)
            X = X.merge(pat_rest.reset_index(drop=True), on=["series_id"], how="left")
            #print(X)
    return X


def construct_X_lstm(database:str, 
                     lookback=7, 
                     aggregated_hours=4, 
                     seed=None, 
                     inc_ab=True, 
                     has_microbiology=False):
    path = 'data/episodes/'+database+'/microbiology_res_'+str(has_microbiology)+'/seed_'+str(seed)
    
    train_lstm_data = convert_to_next_day(path+"/train_lstm_data.parquet")
    validation_lstm_data = convert_to_next_day(path+"/validation_lstm_data.parquet")
    test_data = convert_to_next_day(path+"/test_data.parquet")

    X_lstm_validation = construct_X_lstm_helper(database, validation_lstm_data, lookback_days=lookback, number_of_aggregated_hours=aggregated_hours, inc_ab=inc_ab)
    X_lstm_train = construct_X_lstm_helper(database, train_lstm_data, lookback_days=lookback, number_of_aggregated_hours=aggregated_hours, inc_ab=inc_ab)
    X_lstm_test = construct_X_lstm_helper(database, test_data, lookback_days=lookback, number_of_aggregated_hours=aggregated_hours, inc_ab=inc_ab)

    # fill nan with 0
    X_lstm_train = X_lstm_train.fillna(0)
    X_lstm_validation = X_lstm_validation.fillna(0)
    X_lstm_test = X_lstm_test.fillna(0)

    X_lstm_train = X_lstm_train.merge(train_lstm_data[['series_id','ID']], on=["series_id"], how="inner").sort_values(['series_id','measurement_number'])
    X_lstm_validation = X_lstm_validation.merge(validation_lstm_data[['series_id','ID']], on=["series_id"], how="inner").sort_values(['series_id','measurement_number'])
    X_lstm_test = X_lstm_test.merge(test_data[['series_id','ID']], on=["series_id"], how="inner").sort_values(['series_id','measurement_number'])

    y_lstm_train = train_lstm_data
    y_lstm_validation = validation_lstm_data
    y_lstm_test = test_data

    dest_path = "data/model_input/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)


    # scale X
    if database == 'mimic':
        cols_to_standardize = X_lstm_train.columns.difference(['ID', 'series_id', 'measurement_number'])
        scaler = StandardScaler()
        X_lstm_train[cols_to_standardize] = scaler.fit_transform(X_lstm_train[cols_to_standardize])
        X_lstm_validation[cols_to_standardize] = scaler.transform(X_lstm_validation[cols_to_standardize])
        X_lstm_test[cols_to_standardize] = scaler.transform(X_lstm_test[cols_to_standardize])
        joblib.dump(scaler, dest_path+"/"+"scaler.save")
    else:
        mimic_dest_path = "data/model_input/lstm/mimic/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)
        X_mimic = pd.read_parquet(mimic_dest_path+"/X_lstm_train.parquet")
        scaler = joblib.load(mimic_dest_path+"/"+"scaler.save")

        X_lstm_train[list(set(X_mimic.columns).difference(set(X_lstm_train.columns)))] = 0
        X_lstm_train = X_lstm_train[X_mimic.columns]
        cols_to_standardize = X_lstm_train.columns.difference(['ID', 'series_id', 'measurement_number'])
        X_lstm_train[cols_to_standardize] = scaler.transform(X_lstm_train[cols_to_standardize])

        X_lstm_validation[list(set(X_mimic.columns).difference(set(X_lstm_validation.columns)))] = 0
        X_lstm_validation = X_lstm_validation[X_mimic.columns]
        X_lstm_validation[cols_to_standardize] = scaler.transform(X_lstm_validation[cols_to_standardize])

        X_lstm_test[list(set(X_mimic.columns).difference(set(X_lstm_test.columns)))] = 0
        X_lstm_test = X_lstm_test[X_mimic.columns]
        X_lstm_test[cols_to_standardize] = scaler.transform(X_lstm_test[cols_to_standardize])

        


    X_lstm_train.to_parquet(dest_path+"/"+"X_lstm_train.parquet")
    X_lstm_validation.to_parquet(dest_path+"/"+"X_lstm_validation.parquet")
    X_lstm_test.to_parquet(dest_path+"/"+"X_lstm_test.parquet")

    y_lstm_train.to_parquet(dest_path+"/"+"y_lstm_train.parquet")
    y_lstm_validation.to_parquet(dest_path+"/"+"y_lstm_validation.parquet")
    y_lstm_test.to_parquet(dest_path+"/"+"y_lstm_test.parquet")

