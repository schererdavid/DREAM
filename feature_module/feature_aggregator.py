import json
import pandas as pd
import math
import gc
import os

def sort_dataframe_for_chunking(path):
    df = pd.read_parquet(path)
    df = df.sort_values(['ID'])
    df.to_parquet(path)

    last_ID = df['ID'].iloc[-1]
    #display(df)
    return df.shape, last_ID


def aggregate_to_5min(database:str, method:str):
    feature_path = 'data/features/'+database+'/'
    aggregated_path = 'data/features/'+database+'_5min/'
    
    if not os.path.exists(aggregated_path):
        os.makedirs(aggregated_path)

    # delete already computed files
    for f in os.listdir(aggregated_path):
        os.remove(os.path.join(aggregated_path, f))

    with open(database+'_feature_list.json', 'r') as file:
            features = json.load(file)

    #file_list = get_file_names(directory_path)
    file_list = sorted(list(features.keys()))

    for f in file_list:
        print(f)
        # first make sure the IDs are sorted
        shape, last_ID = sort_dataframe_for_chunking(feature_path+f)
        #print("lastid", last_ID)

        # todo operate with chunks
        chunksize = 1000000
        max = math.ceil(last_ID/chunksize) 
        chunk_count = 0
        left_over = pd.DataFrame()
        #print('max', max)
        for i in range(0, max):
            chunk_count += 1
            #print('chunk_count',chunk_count)

            #print('bounds', i*chunksize, (i+1)*chunksize)
            df = pd.read_parquet(feature_path+f, filters=[('ID', '>=', i*chunksize), ('ID', '<=', (i+1)*chunksize)])
            #display(df)
            
            df = pd.concat([left_over, df])

            
            if not chunk_count == (max-1):
                if df.shape[0] != 0:
                    current_last_ID = df['ID'].iloc[-1]
                    df = df[df['ID'] != current_last_ID]
                    left_over = df[df['ID'] == current_last_ID].copy()
                else:
                    left_over = pd.DataFrame()

            if 'Time' in df.columns:
                df['Time'] = df['Time'].dt.floor('5min')  
                if method == 'mean':
                    agg_functions = {
                        list(df.columns)[2]: ['mean', 'count'],
                    }
                elif method == 'median':
                    agg_functions = {
                        list(df.columns)[2]: ['median', 'count'],
                    }
                else:
                    raise Exception("method not supported")
                
                result_df = df.groupby(['ID', 'Time']).agg(agg_functions)
                result_df.columns = ['_'.join(col) for col in result_df.columns]
                result_df = result_df.reset_index()


                filname = f.split(".")[0]+'.parquet'
                if not os.path.exists(aggregated_path+filname):
                    result_df.to_parquet(aggregated_path+filname, engine='fastparquet')
                else:
                    result_df.to_parquet(aggregated_path+filname, engine='fastparquet', append=True) 
            
            del df
            gc.collect()
