import os
import pandas as pd
import json
import gc

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names


def repeat_every_hosp_day(df: pd.DataFrame, concepts_path='data/concepts/eicu/', constant_fact_column='Quantity'):
    admin_case = pd.read_parquet(concepts_path+'AdministrativeCase.parquet')
    admin_case = admin_case[admin_case['Location'] == 'hosp']

    merged = pd.merge(admin_case, df, how='inner', on='SubjectPseudoIdentifier')
    merged['AdmissionDateTime'] = pd.to_datetime(merged['AdmissionDateTime'])
    merged['DischargeDateTime'] = pd.to_datetime(merged['DischargeDateTime'])
    merged['DateRange'] = merged.apply(lambda row: pd.date_range(start=row['AdmissionDateTime'], end=row['DischargeDateTime'], freq='D'), axis=1)

    exploded = merged.explode('DateRange')

    result_df = exploded[['SubjectPseudoIdentifier', 'DateRange', constant_fact_column]].copy()
    result_df.rename(columns={'DateRange': 'Time'}, inplace=True)

    return result_df


def extract_features(database: str):
    concepts_path = 'data/concepts/'+database+'/'
    features_path = 'data/features/'+database+'/'
    
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    concept_list = get_file_names(concepts_path)

    # we do not use the birth date and the death date directly as features
    concept_list.remove('BirthDate.parquet')
    concept_list.remove('DeathDate.parquet')

    features = {}
    # from every concept we extract the features which we will use    
    for concept_file in concept_list:

        #if concept_file != 'BodyHeight.parquet':
        #   continue
        print(concept_file)
        
        #if concept_file == 'HeartRate.parquet':
        #else:
        
        df = pd.read_parquet(concepts_path+concept_file)
        concept = concept_file.split('.')[0]

        #concept = 'RespiratoryRate'

        print("Concept:", concept)
        
        if concept == 'BodyMassIndex':
            if database == 'eicu':
                df = repeat_every_hosp_day(df, concepts_path, 'Quantity')
            df.rename(columns={'SubjectPseudoIdentifier':'ID','DeterminationDateTime':'Time', 'Quantity':'BMI'}, inplace=True)
            df.to_parquet(features_path+"bmi.parquet")
            features["bmi.parquet"] = {'type':'numerical', 'aggregated_hours':24}
        elif concept == 'BodyTemperature':
            df.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Temperature'}, inplace=True)
            df.to_parquet(features_path+"temperature.parquet")
            features["temperature.parquet"] = {'type':'numerical', 'aggregated_hours':4}
        elif concept == 'DrugAdministrationEvent':
            # we only use the fact of applied ABs
            df.rename(columns={'SubjectPseudoIdentifier':'ID','StartDateTime':'Time'}, inplace=True)
            df['AB_applied'] = 1
            df[['ID','Time','AB_applied']].to_parquet(features_path+"AB_applied.parquet")
            features["AB_applied.parquet"] = {'type':'numerical', 'aggregated_hours':12}
        elif concept == 'LabResult':
            # Microbiology
            df.rename(columns={'SubjectPseudoIdentifier':'ID'}, inplace=True)
            if database == 'mimic':
                blood_cultures = df[(df['LabTest'] == 'BLOOD CULTURE') & (df['QualitativeResult'] != 'CANCELLED')].copy()
                blood_cultures['isPositive'] = blood_cultures['QualitativeResult'].notnull()
            elif database == 'eicu':
                blood_cultures = df[df['LabTest'].isin(['Blood, Venipuncture', 'Blood, Central Line'])].copy()
                blood_cultures['isPositive'] = blood_cultures['QualitativeResult'] != 'no growth'
            elif database == 'pic':
                blood_cultures = df[df['LabTest'].str.contains('LIS0162')].copy()
                blood_cultures['isPositive'] = blood_cultures['QualitativeResult'].notnull()
            else:
                raise Exception("database not supported")
            # assume blood culutre results are available after 2 days
            blood_cultures['ReportDateTime'] = blood_cultures['CollectionDateTime']+ pd.Timedelta(days=2)
            blood_cultures['BloodcultureCollected'] = 1
            # save collection feature
            blood_cultures.rename(columns={'CollectionDateTime':'Time'}, inplace=True)
            blood_cultures[['ID','Time','BloodcultureCollected']].to_parquet(features_path+"BloodcultureCollected.parquet")
            features["BloodcultureCollected.parquet"] = {'type':'numerical', 'aggregated_hours':12}
            blood_cultures.drop(['Time'], axis=1, inplace=True)
            # save result feature
            blood_cultures.rename(columns={'ReportDateTime':'Time'}, inplace=True)
            blood_cultures = blood_cultures[blood_cultures['Time'].notnull()]
            blood_cultures[['ID','Time','isPositive']].to_parquet(features_path+"BloodcultureResult.parquet")
            features["BloodcultureResult.parquet"] = {'type':'numerical', 'aggregated_hours':12}
            
            # Lab Events
            labs = df[df['QuantitativeResult'].notna()].copy()
            labs['LabCollected'] = 1
            labtests = labs['LabTest'].unique()
            # collection feature
            for labtest in labtests:
                test = labs[labs['LabTest'] == labtest].copy()
                test.rename(columns={'CollectionDateTime':'Time', 'LabCollected':'lab_collected_'+str(labtest)}, inplace=True)
                test[['ID','Time','lab_collected_'+str(labtest)]].to_parquet(features_path+'lab_collected_'+str(labtest)+".parquet")
                features['lab_collected_'+str(labtest)+".parquet"] = {'type':'numerical', 'aggregated_hours':12}
            # result feature
            for labtest in labtests:
                test = labs[labs['LabTest'] == labtest].copy()
                test.rename(columns={'ReportDateTime':'Time', 'QuantitativeResult':'lab_result_'+str(labtest)}, inplace=True)
                test[['ID','Time','lab_result_'+str(labtest)]].to_parquet(features_path+'lab_result_'+str(labtest)+".parquet")
                features['lab_result_'+str(labtest)+".parquet"] = {'type':'numerical', 'aggregated_hours':12}
        elif concept == 'HeartRate':
            df.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Heartrate'}, inplace=True)
            df.to_parquet(features_path+"heartrate.parquet")
            features["heartrate.parquet"] = {'type':'numerical', 'aggregated_hours':4}
        elif concept == 'OxygenSaturation':
            df.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'OxygenSaturation'}, inplace=True)
            df.to_parquet(features_path+"oxygenSaturation.parquet")
            features["oxygenSaturation.parquet"] = {'type':'numerical', 'aggregated_hours':4}
        elif concept == 'RespiratoryRate':
            df.rename(columns={'SubjectPseudoIdentifier':'ID','DeterminationDateTime':'Time', 'Quantity':'RespiratoryRate'}, inplace=True)
            df.to_parquet(features_path+"respiratoryRate.parquet")
            features["respiratoryRate.parquet"] = {'type':'numerical', 'aggregated_hours':4}
        elif concept == 'BodyHeight':
            if database == 'eicu':
                df = repeat_every_hosp_day(df, concepts_path, 'Quantity')
            df.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Height (cm)'}, inplace=True)
            df.to_parquet(features_path+"height.parquet")
            features["height.parquet"] = {'type':'numerical', 'aggregated_hours':24}

            if database == 'pic':
                # also construct and add bmi
                df_weight = pd.read_parquet(concepts_path+'BodyWeight.parquet')
                df_weight.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Weight (kg)'}, inplace=True)
                df['month'] = df['Time'].dt.month
                df['year'] = df['Time'].dt.year
                df.drop(['Time'], axis=1, inplace=True)
                df_weight['month'] = df_weight['Time'].dt.month
                df_weight['year'] = df_weight['Time'].dt.year
                df = pd.merge(df, df_weight, how='inner', on=['ID','month','year'])
                
                df.dropna(subset=['Weight (kg)','Height (cm)'], inplace=True)
                df = df[(df['Weight (kg)'] != 0) & (df['Height (cm)'] != 0)]
                df['BMI'] = df['Weight (kg)'] / ((df['Height (cm)'] / 100) ** 2)
                df.drop(['month','year','Weight (kg)','Height (cm)'], axis=1, inplace=True)
                df.to_parquet(features_path+"bmi.parquet")
                features["bmi.parquet"] = {'type':'numerical', 'aggregated_hours':24}
        elif concept == 'Ethnicity':
            df.rename(columns={'SubjectPseudoIdentifier':'ID'}, inplace=True)
            one_hot_encoded_df = pd.get_dummies(df, columns=['Ethnicity'])
            for ethnicity in one_hot_encoded_df.columns[1:]:
                name_cleaned = ethnicity.replace(" ", "").replace("/", "").replace("\\", "").replace("-", "")
                one_hot_encoded_df[['ID',ethnicity]].to_parquet(features_path+name_cleaned+".parquet")
                features[name_cleaned+".parquet"] = {'type':'constant', 'aggregated_hours':24}
        elif concept == 'BodyWeight':
            if database == 'eicu':
                df = repeat_every_hosp_day(df, concepts_path, 'Quantity')
            df.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Weight (kg)'}, inplace=True)
            df.to_parquet(features_path+"weight.parquet")
            features["weight.parquet"] = {'type':'numerical', 'aggregated_hours':24}
        elif concept == 'BloodPressure':
            systolic = df[df['Type'] == 'systolic'].copy()
            diastolic = df[df['Type'] == 'diastolic'].copy()
            systolic.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Systolic BP'}, inplace=True)
            diastolic.rename(columns={'SubjectPseudoIdentifier':'ID','MeasurementDateTime':'Time', 'Quantity':'Diastolic BP'}, inplace=True)
            systolic.drop(['Type'], inplace=True, axis=1)
            diastolic.drop(['Type'], inplace=True, axis=1)
            systolic.to_parquet(features_path+"systolic.parquet")
            features["systolic.parquet"] = {'type':'numerical', 'aggregated_hours':4}
            diastolic.to_parquet(features_path+"diastolic.parquet")
            features["diastolic.parquet"] = {'type':'numerical', 'aggregated_hours':4}
        elif concept == 'AdministrativeCase': # maybe just add the age at admission later based on birth day
            # calculate age at hospitalization 
            birthDateTime = pd.read_parquet(concepts_path+'BirthDate.parquet')
            df = df[df['Location'] == 'hosp'].copy()
            merged = pd.merge(df, birthDateTime, how='inner', on='SubjectPseudoIdentifier')
            merged['ageAtAdmission'] = (merged['AdmissionDateTime'] - pd.to_datetime(merged['BirthDateTime'], format='%Y')).dt.total_seconds() / (365.25 * 24 * 60 * 60)  
            merged['Time'] = merged.apply(lambda row: pd.date_range(start=pd.to_datetime(row['AdmissionDateTime']),
                                                                    end=pd.to_datetime(row['DischargeDateTime']), freq='D'), axis=1)
            result_df = merged.explode('Time')[['SubjectPseudoIdentifier', 'Time', 'ageAtAdmission']]
            result_df.rename(columns={'SubjectPseudoIdentifier': 'ID'}, inplace=True)
            result_df[['ID', 'Time', 'ageAtAdmission']].to_parquet(features_path + "ageAtAdmission.parquet")
            features["ageAtAdmission.parquet"] = {'type': 'numerical', 'aggregated_hours': 24}
        elif concept == 'AdministrativeGender':
            df.rename(columns={'SubjectPseudoIdentifier':'ID'}, inplace=True)
            df['isMale'] = df['Code'] == 'M'
            df[['ID','isMale']].to_parquet(features_path+"gender.parquet")
            features["gender.parquet"] = {'type':'constant', 'aggregated_hours':24}
        else:
            raise Exception('Concept not supported')
        
        del df
        gc.collect()
        
    print("save json...")    
    with open(database+'_feature_list.json', 'w') as file:
        json.dump(features, file)
        print("json saved") 