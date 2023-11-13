import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.model_selection import train_test_split


def get_hosp_icu_stays(concepts_path:str):
    """Computes a dataframe of all icu stays with their corresponding hospital stay for each patient

    Args:
        concepts_path: path to the stored concepts

    Returns:
        df: a dataframe containing all icu stays with their corresponding hospital stay for each patient
        
    """

    print("---get all stays---") 
    stays = pd.read_parquet(concepts_path+"/AdministrativeCase.parquet").rename(columns={'SubjectPseudoIdentifier':'ID'})
    print("shape stays", stays.shape)
    #stays['AdmissionDateTime'] = stays['AdmissionDateTime'].dt.floor('5min')  
    #stays['DischargeDateTime'] = stays['DischargeDateTime'].dt.floor('5min')  
    icu_stays = stays[stays['Location'] == 'icu'].copy()
    hosp_stays = stays[stays['Location'] == 'hosp'].copy()


    print("---assign icu stays to hospital stays---")
    df = pd.merge(icu_stays,hosp_stays,how='inner',on='ID', suffixes=('_icu', '_hosp'))
    df = df[(df['AdmissionDateTime_hosp'] <= df['AdmissionDateTime_icu']) & (df['DischargeDateTime_icu'] <= df['DischargeDateTime_hosp'])]
    df.drop(['Location_icu','Location_hosp'],axis=1,inplace=True)
    return df

def add_deathtime(concepts_path:str, df:pd.DataFrame):
    deathtimes = pd.read_parquet(concepts_path+"/DeathDate.parquet").rename(columns={'SubjectPseudoIdentifier':'ID'})
    deathtimes.drop_duplicates(inplace=True)

    count_nat = deathtimes['DeathDateTime'].isna().sum()
    count_days = deathtimes['DeathDateTime'].apply(lambda x: isinstance(x, pd.Timestamp) and x.time() == datetime.min.time()).sum()
    count_date_times = deathtimes['DeathDateTime'].apply(lambda x: isinstance(x, pd.Timestamp) and x.time() != datetime.min.time()).sum()

    # print("Anzahl NaT:", count_nat)
    # print("Anzahl ganze Tage:", count_days)
    # print("Anzahl DateTime-Werte:", count_date_times)

    # we only take the deathdatetimes wehere we have the exact one
    deathtimes = deathtimes[deathtimes['DeathDateTime'].apply(lambda x: isinstance(x, pd.Timestamp) and x.time() != datetime.min.time())]

    t = df.copy()
    temp = pd.merge(t,deathtimes,how='left',on='ID')

    # set deathtimes to NaT if it is not withing the hospital stay
    temp.loc[~((temp['DeathDateTime'] >= temp['AdmissionDateTime_hosp']) & (temp['DeathDateTime'] <= temp['DischargeDateTime_hosp'])), 'DeathDateTime'] = pd.NaT

    return temp

# (note: for a patient with multiple icu stays within the same hospital stay, a treatment can be listet multiple times)
def check_time(row):
    if row['Time'] < row['AdmissionDateTime_icu']:
        return 'pre'
    elif row['Time'] > row['DischargeDateTime_icu']:
        return 'post'
    elif row['AdmissionDateTime_icu'] <= row['Time'] and row['Time'] <= row['DischargeDateTime_icu']:
        return 'in'
    else:
        raise Exception("not possible")

def attach_ab_treatments_to_hosp_icu_stays(hosp_icu_stays, features_path):
    ab_applications = pd.read_parquet(features_path+"/AB_applied.parquet")
    print('shape ab_applications', ab_applications.shape)
    df = pd.merge(hosp_icu_stays,ab_applications,how='inner',on='ID')
    # filter to get all applications within the hospital stay
    df = df[(df['AdmissionDateTime_hosp'] <= df['Time']) & (df['Time'] <= df['DischargeDateTime_hosp'])].copy()
    print("---check if treatment was before, after or within an icu admission---")
    df['pre_in_post_icu'] = df.apply(check_time, axis=1)
    return df

def add_labels(episodes):
    episodes['lot'] = episodes['endtime'] - episodes['starttime'] + pd.Timedelta(1, 'days')
    episodes['lot_in_days'] = episodes['lot'].apply(lambda x: x.total_seconds() / (3600 * 24))
    episodes.drop(['Time','first','last', 'difference', 'hours_between_applications'], axis=1, inplace=True)
    episodes = episodes.drop_duplicates()
    return episodes


def extract_episodes(df, min_hours_new_episode):
    df = df[['ID','AdmissionDateTime_hosp','Time']].copy().drop_duplicates()
    df = df.sort_values(['ID', 'AdmissionDateTime_hosp', 'Time'])
    df['first'] = df.groupby(['ID', 'AdmissionDateTime_hosp'])['Time'].transform(lambda x: x == x.min())
    df['last'] = df.groupby(['ID', 'AdmissionDateTime_hosp'])['Time'].transform(lambda x: x == x.max()) # this line may not be needed
    # calculate the differences between all the applications
    df['difference'] = df.groupby(['ID', 'AdmissionDateTime_hosp'])['Time'].diff()
    # mark the ones where the pause was larger than "min_hours_new_episode" also as first
    df.loc[df['difference'] > pd.Timedelta(hours=min_hours_new_episode), 'first'] = True
    # mark the application before first as last
    df['last'] = df['first'].shift(-1)
    df.loc[df.index[-1], 'last'] = True
    # calculate the number of hours between the applications
    df['hours_between_applications'] = df['difference'].apply(lambda x: x.total_seconds() / 3600)
    # all one time treatments
    onetime = df[(df['first'] == True) & (df['last'] == True)].copy()
    onetime['endtime'] = onetime['Time'] 
    onetime['starttime'] = onetime['Time']
    # merge start and end of treatments together
    starts = df[(df['first'] == True) & (df['last'] == False)].copy()
    starts['starttime'] = starts['Time'] 
    ends = df[(df['first'] == False) & (df['last'] == True)].copy()
    ends['endtime'] = ends['Time'] 
    repeated_treatment = pd.concat([starts.reset_index(drop=True), ends[['endtime']].reset_index(drop=True)], axis=1)
    # merge one time treatments with the rest together
    df = pd.concat([onetime, repeated_treatment], axis=0)
    return df


def add_location(episodes, ab_hosp_icu):
    treatment_location = pd.merge(episodes.copy(), ab_hosp_icu[['ID','AdmissionDateTime_hosp','DischargeDateTime_hosp','AdmissionDateTime_icu','DischargeDateTime_icu']].drop_duplicates().copy(), how='inner', on=['ID','AdmissionDateTime_hosp'])
    treatment_location['episode_location']  = treatment_location.apply(locate, axis=1)
    #print('all icu',treatment_location[treatment_location['episode_location'] == 'all icu'].shape[0])
    #print('started in icu, ended outside',treatment_location[treatment_location['episode_location'] == 'started in icu, ended outside'].shape[0])
    #print('not handled',treatment_location[treatment_location['episode_location'] == 'not handled'].shape[0])
    #if treatment_location[treatment_location['episode_location'] == 'not handled'].shape[0] > 0:
    #    print(treatment_location[treatment_location['episode_location'] == 'not handled'])
    return treatment_location

# FILTER WHICH EPISODES TO KEEP (ones which start in icu etc.)
def locate(row):
    if row['AdmissionDateTime_icu'] <= row['starttime'] and row['endtime'] <= row['DischargeDateTime_icu']:
        return 'all icu'
    elif row['AdmissionDateTime_icu'] <= row['starttime'] and row['starttime'] <= row['DischargeDateTime_icu'] and row['DischargeDateTime_icu'] <= row['endtime']:
        return 'started in icu, ended outside'
    elif row['starttime'] <= row['AdmissionDateTime_icu'] and row['endtime'] <= row['AdmissionDateTime_icu']:
        return 'started before icu, ended before icu'
    elif row['starttime'] <= row['AdmissionDateTime_icu'] and row['AdmissionDateTime_icu'] <= row['endtime'] and row['endtime'] <= row['DischargeDateTime_icu']:
        return 'started before icu, ended in icu'
    elif row['DischargeDateTime_icu'] <= row['starttime'] and row['DischargeDateTime_icu'] <= row['endtime']:
        return 'started after icu, ended after icu'
    elif row['starttime'] <= row['AdmissionDateTime_icu'] and row['DischargeDateTime_icu'] <= row['endtime']:
        return 'started before icu, ended after icu'
    else:
        return 'not handled'

def filter_episodes(database:str, treatment_location:str):
    if database == 'mimic':
        episodes_of_interest = treatment_location[treatment_location['episode_location'].isin(['all icu','started in icu, ended outside'])]
    elif database == 'eicu':
        episodes_of_interest = treatment_location[treatment_location['episode_location'].isin(['all icu','started in icu, ended outside'])]
    elif database == 'pic':
        episodes_of_interest = treatment_location[treatment_location['episode_location'].isin(['all icu','started in icu, ended outside'])]
    else:
        raise Exception("database not supported")


    # filter the ones where end was not observable
    # we don't do that anymore since we also want to use the right censored episodes
    #episodes_of_interest = episodes_of_interest[episodes_of_interest['starttime'] + pd.Timedelta(5, 'days') <= episodes_of_interest['DischargeDateTime_hosp']]
    
    return episodes_of_interest

def add_endtime_censored(df):
    df = df.copy()
    df['censored'] = df['endtime'] + pd.Timedelta(2, 'days') > df['DischargeDateTime_hosp']
    return df


def add_microbiology(episodes_of_interest, features_path):
    # add the microbiology results
    micro = pd.read_parquet(features_path+"/BloodcultureResult.parquet")
    temp = pd.merge(episodes_of_interest, micro, how='left', on='ID')

    # a test was made in the respective episode
    tests_in_episode = temp[(temp['starttime'] - pd.Timedelta(2, 'days') <= temp['Time']) & (temp['Time'] <= temp['endtime'])].drop(['Time'],axis=1).drop_duplicates().copy()

    # look if at least one test was positive
    tests_in_episode['isPositive'] = tests_in_episode.groupby(list(tests_in_episode.columns[0:-1]))['isPositive'].transform('any')
    tests_in_episode = tests_in_episode.drop_duplicates()

    tests_in_episode = pd.merge(episodes_of_interest, tests_in_episode, how='left', on=list(tests_in_episode.columns[0:-1]))

    return tests_in_episode

def construct_episodes(database:str, only_with_microbiology_res=True) -> pd.DataFrame:
    concepts_path = "data/concepts/"+database
    features_path = "data/features/"+database

    print("---MAP ANTIBIOTICS APPLICATIONS TO HOSPITAL STAYS---")

    hosp_icu_stays = get_hosp_icu_stays(concepts_path)
    print('shape hosp_icu_stays', hosp_icu_stays.shape)

    print("---add antibiotic treatments---") 

    ab_treatments_hosp_icu_stays = attach_ab_treatments_to_hosp_icu_stays(hosp_icu_stays, features_path)
    print('shape ab_treatments_hosp_icu_stays', ab_treatments_hosp_icu_stays.shape)

    # copy for later use
    ab_hosp_icu = ab_treatments_hosp_icu_stays.copy()

    print("--- COMPUTE AND EXTRACT ALL POSSIBLE EPISODES ----")
    episodes = extract_episodes(ab_treatments_hosp_icu_stays, 48)
    print('shape episodes', episodes.shape)

    print("--- ADD LABELS ----")
    episodes = add_labels(episodes)

    # construct all possible episode
    print("--- ADD Location ----")
    treatment_location = add_location(episodes, ab_hosp_icu)

    print("--- episodes filtered ---")
    episodes_of_interest = filter_episodes(database=database, treatment_location=treatment_location)
    print('shape episodes_of_interest', episodes_of_interest.shape)

    episodes_of_interest = episodes_of_interest.drop_duplicates()
    episodes_of_interest

    print("-- ADD THE MICROBIOLOGY RESULTS --") 
    ep = add_microbiology(episodes_of_interest, features_path)

    ep = add_deathtime(concepts_path, ep)
    
    # filter that we only have the episodes where a blood culture was made
    if only_with_microbiology_res:
        print("ATTENTION: only patients with a microbiology result were included")
        ep = ep[ep['isPositive'].notna()]
    else:
        print("ATTENTION: patients without a microbiology result were included")


    counts = ep['episode_location'].value_counts()

    ep = add_endtime_censored(ep)

    path = 'data/episodes/'+database+'/microbiology_res_'+str(only_with_microbiology_res)
    
    if not os.path.exists(path):
        os.makedirs(path)

    ep.to_parquet(path+'/all_episodes.parquet')

    print("episodes shape:", ep.shape)




def split_data_by_ids(sourname, destination1, destination2, path, test_size=0.15, random_state=None):
    df = pd.read_parquet(sourname)

    unique_ids = df['ID'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_data = df[df['ID'].isin(train_ids)]
    test_data = df[df['ID'].isin(test_ids)]

    if not os.path.exists(path+'/seed_'+str(random_state)):
        os.makedirs(path+'/seed_'+str(random_state))

    train_data.to_parquet(destination1)
    test_data.to_parquet(destination2)
    