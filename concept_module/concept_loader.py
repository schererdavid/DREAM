
import pandas as pd
import os
from tqdm import tqdm

def load_concepts(database: str):
    
    # check from which database we are loading data
    if database == 'mimic':
        from concept_module.table_to_concept import mimic_table_to_concept
        load_mimic_concepts(mimic_table_to_concept)
    elif database == 'eicu':
        from concept_module.table_to_concept import eicu_table_to_concept
        load_eicu_concepts(eicu_table_to_concept)
    elif database == 'pic':
        from concept_module.table_to_concept import pic_table_to_concept
        load_pic_concepts(pic_table_to_concept)
    else:
        raise Exception('database not supported')
    
def load_eicu_concepts(table_to_concept_dict: dict):
    path = 'data/concepts/eicu/'
    if not os.path.exists(path):
        os.makedirs(path)

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    # we handle each table seperately
    for table in table_to_concept_dict:
        print(table)

        # get all concepts associated to the specific table
        concepts = table_to_concept_dict[table]['concepts']
        print(concepts)

        # get the columns and their data types which we are interested in this table
        columns = list(table_to_concept_dict[table]['columns'].keys())

        
        # get non date columns
        dtype_without_dates = {key: value for key, value in table_to_concept_dict[table]['columns'].items()}
        #display(dtype_without_dates)


        # if table != "medication.csv.gz":
        #    continue

        # load the table in chunks
        df_chunks = pd.read_csv('data/raw/eicu/'+table, usecols=columns, chunksize=1000000, dtype=dtype_without_dates)

        # we divide the loaded table into chunks
        for chunk in tqdm(df_chunks):
            #display(chunk)
            # for each concept which this table has information for, we extract the relevant data and add it to the concept file
            for concept in concepts:
                # if concept != 'DrugAdministrationEvent': #'RespiratoryRate':
                #     continue
                
                if concept == 'AdministrativeCase':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier'})
                        # icu
                        df['AdmissionDateTime'] = pd.to_datetime('2000-01-01 00:00')
                        df['DischargeDateTime'] = df['AdmissionDateTime'] + pd.to_timedelta(df['unitdischargeoffset'], unit="minutes") 
                        df['Location'] = 'icu'
                        # hosp
                        temp = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier'})
                        temp['AdmissionDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['hospitaladmitoffset'], unit="minutes") 
                        temp['DischargeDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['hospitaldischargeoffset'], unit="minutes")
                        temp['Location'] = 'hosp'

                        df = pd.concat([df, temp], axis=0)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','AdmissionDateTime','DischargeDateTime','Location']]
                    df = df.drop_duplicates()
                elif concept == 'AdministrativeGender':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier',
                                                          'gender':'Code'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Code']]
                    df['Code'] = df['Code'].replace({'Male': 'M', 'Female': 'F'})
                    df = df[df['Code'].isin(['M', 'F'])]
                elif concept == 'BodyWeight':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 'admissionweight':'Quantity'})
                        df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00')
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                    df = df[df['Quantity'].notna()]
                elif concept == 'BodyHeight':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 'admissionheight':'Quantity'})
                        df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00')
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                    df = df[df['Quantity'].notna()]
                elif concept == 'BodyMassIndex':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier'})
                        # calculation of BMI
                        df = df.dropna(subset=['admissionheight', 'admissionweight'])
                        df = df[df['admissionheight'] != 0]
                        df['admissionheight_m'] = df['admissionheight'] / 100
                        df['Quantity'] = df['admissionweight'] / (df['admissionheight_m'] ** 2)
                        df['DeterminationDateTime'] = pd.to_datetime('2000-01-01 00:00')
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','DeterminationDateTime','Quantity']]
                    df = df[df['Quantity'].notna()]
                elif concept == 'BirthDate':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier'})
                        df['age'] = df['age'].replace({'> 89': '90'})
                        df['age'] = df['age'].astype(float)
                        df = df[df['age'].notna()]

                        df['age'] = df['age'] * pd.to_timedelta(365.25, unit='D')

                        df['BirthDateTime'] = pd.to_datetime('2000-01-01 00:00') 

                        df['BirthDateTime'] = df['BirthDateTime'] - df['age']
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','BirthDateTime']]
                elif concept == 'DeathDate':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier'})
                        
                        df = df[df['hospitaldischargestatus'] == 'Expired']

                        def calculate_deathdate(row):
                            if row['unitdischargestatus'] == 'Expired':
                                return min(row['hospitaldischargeoffset'], row['unitdischargeoffset'])
                            else:
                                return row['hospitaldischargeoffset']

                        df['DeathDateTime'] = df.apply(calculate_deathdate, axis=1)
                        df['DeathDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['hospitaldischargeoffset'], unit="minutes")
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','DeathDateTime']]

                elif concept == 'Ethnicity':
                    if table == 'patient.csv.gz':
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 'ethnicity':'Ethnicity'})
                        df['Ethnicity'] = df['Ethnicity'].replace({'Caucasian': 'WHITE', 
                                                                   'African American': 'BLACK/AFRICAN AMERICAN',
                                                                   'Hispanic': 'HISPANIC OR LATINO',
                                                                   'Asian': 'ASIAN',
                                                                   'Native American' : 'AMERICAN INDIAN/ALASKA NATIVE'
                                                                   })
                        df = df[(~df['Ethnicity'].isin(['Other/Unknown'])) & df['Ethnicity'].notna()]
                    else:
                        raise Exception(f"Table {table} not supported")    

                    df = df[['SubjectPseudoIdentifier','Ethnicity']]
                elif concept == 'LabResult':
                    if table == "microLab.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'culturesite':'LabTest', 
                                                          'organism':'QualitativeResult',
                                                          'culturetakenoffset': 'CollectionDateTime'})
                        df['ReportDateTime'] = pd.to_datetime(float('NaN'))
                        df['QuantitativeResult'] = float('NaN')
                        df['CollectionDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['CollectionDateTime'], unit="minutes")
                        df = df[df['LabTest'].isin(['Blood, Venipuncture', 'Blood, Central Line'])].copy()
                    elif table == "lab.csv.gz" :
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'labname':'LabTest', 
                                                          'labresulttext':'QualitativeResult', 
                                                          'labresult':'QuantitativeResult',
                                                          'labresultoffset': 'CollectionDateTime',
                                                          'labresultrevisedoffset': 'ReportDateTime'})
                        #display(df)
                        # unfortunately we can't handle all labs at once so we have to make a selection
                        labs = []
                        from constant_files.lab_list import lab_list_eicu
                        for key in lab_list_eicu:
                            labs = labs + lab_list_eicu[key]

                        df = df[df['LabTest'].isin(labs)].copy()


                        # instead of the itemid, we use the name of the lab test
                        def find_key(dictionary, integer):
                            for key, value_list in dictionary.items():
                                if integer in value_list:
                                    return key
                            return None
                        
                        df['LabTest'] = df['LabTest'].astype(str)
                        for itemid in labs:
                            lab_test_name = find_key(lab_list_eicu, itemid)
                            df.loc[df['LabTest'] == str(itemid), 'LabTest'] = lab_test_name
                        df['CollectionDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['CollectionDateTime'], unit="minutes")
                        df['ReportDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['ReportDateTime'], unit="minutes")
                        df = df[df['QuantitativeResult'].notna()]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'LabTest', 'CollectionDateTime', 'ReportDateTime', 'QualitativeResult', 'QuantitativeResult']]
                elif concept == 'DrugAdministrationEvent':
                    if table == "medication.csv.gz":
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'drugname': 'Drug', 
                                                          'drugstartoffset':'StartDateTime',
                                                          'drugstopoffset':'EndDateTime'})
                        # only non cancelled drugs
                        df = df[df['drugordercancelled'] == 'No']

                        drugs = pd.read_csv("constant_files/eicu_medication.csv")
                        df = df[(df['drughiclseqno'].isin(list(drugs['drughiclseqno']))) | ((df['gtc'].isin(list(drugs['gtc']))))]

                        df['StartDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['StartDateTime'], unit="minutes")
                        df['EndDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['EndDateTime'], unit="minutes")

                        df = df[df['StartDateTime'] <= df['EndDateTime']]

                        df = df[~df['Drug'].isna()]


                        # we repeat the medication fact every day
                        #display(df.head(50))
                        dfs = []
                        
                        for _, row in df.iterrows():
                            admission_date = pd.to_datetime(row['StartDateTime'])
                            discharge_date = pd.to_datetime(row['EndDateTime'])
                            days = (discharge_date - admission_date).days + 1

                            # start and end
                            subject_df = pd.DataFrame({
                                'SubjectPseudoIdentifier': [row['SubjectPseudoIdentifier']] * 2,
                                'Drug': [row['Drug']] * 2,
                                'StartDateTime': [row['StartDateTime'], row['EndDateTime']]   
                            })
                            dfs.append(subject_df)

                            # in between
                            r = pd.date_range(start=admission_date, end=discharge_date, inclusive='neither', freq='D')
                            if len(r) != 0:
                                subject_df = pd.DataFrame({
                                    'SubjectPseudoIdentifier': [row['SubjectPseudoIdentifier']] * len(r),
                                    'Drug': [row['Drug']] * len(r),
                                    'StartDateTime': r
                                })
                                dfs.append(subject_df)
                        df = pd.concat(dfs, ignore_index=True)

                        #display(result_df.head(50))
                    elif table == "infusionDrug.csv.gz":
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'drugname': 'Drug', 
                                                          'infusionoffset':'StartDateTime'})
                        drugs = pd.read_csv("constant_files/eicu_infusion_drug.csv")
                        df = df[(df['Drug'].isin(list(drugs['drugname'])))]
                        df['StartDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['StartDateTime'], unit="minutes")
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Drug','StartDateTime']] 
                elif concept == 'BloodPressure':
                    if table == "nurseCharting.csv.gz":  
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'nursingchartentryoffset': 'MeasurementDateTime', 
                                                          'nursingchartvalue':'Quantity'})
                        
                        df = df[df['nursingchartcelltypevallabel'].isin(['Invasive BP','Non-Invasive BP'])]

                        df_sys = df[df['nursingchartcelltypevalname'].isin(['Invasive BP Systolic','Non-Invasive BP Systolic'])].copy()
                        df_sys['Type'] = 'systolic'

                        df_dia = df[df['nursingchartcelltypevalname'].isin(['Invasive BP Diastolic','Non-Invasive BP Diastolic'])].copy()
                        df_dia['Type'] = 'diastolic'

                        df = pd.concat([df_sys, df_dia])
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    elif table == "vitalPeriodic.csv.gz":
                        # checked
                        df_sys = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'MeasurementDateTime', 
                                                          'systemicsystolic':'Quantity'})
                        df_sys['Type'] = 'systolic'
                        df_dia = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'MeasurementDateTime', 
                                                          'systemicdiastolic':'Quantity'})
                        df_dia['Type'] = 'diastolic'
                        df = pd.concat([df_sys, df_dia])
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Type','Quantity']]
                    df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['MeasurementDateTime'], unit="minutes")
                elif concept == 'HeartRate':
                    if table == "nurseCharting.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'nursingchartentryoffset':'MeasurementDateTime', 
                                                          'nursingchartvalue': 'Quantity'})
                        df = df[df['nursingchartcelltypevallabel'] == 'Heart Rate']
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    elif table == "vitalPeriodic.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'MeasurementDateTime', 
                                                          'heartrate':'Quantity'})
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]  
                    df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['MeasurementDateTime'], unit="minutes")
                elif concept == 'OxygenSaturation':
                    if table == "nurseCharting.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'nursingchartentryoffset':'MeasurementDateTime', 
                                                          'nursingchartvalue': 'Quantity'})
                        df = df[(df['nursingchartcelltypevallabel'] == 'O2 Saturation') | (df['nursingchartcelltypevallabel'] == 'SpO2')]
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    elif table == "vitalPeriodic.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'MeasurementDateTime', 
                                                          'sao2':'Quantity'})
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float).round().astype(int)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                    df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['MeasurementDateTime'], unit="minutes")
                elif concept == 'RespiratoryRate':
                    if table == "nurseCharting.csv.gz":

                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'nursingchartentryoffset':'DeterminationDateTime', 
                                                          'nursingchartvalue': 'Quantity'})
                        df = df[df['nursingchartcelltypevalname'] == 'Respiratory Rate']
                        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].round().astype(int)
                    elif table == "vitalPeriodic.csv.gz":
                        
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'DeterminationDateTime', 
                                                          'respiration':'Quantity'})
                        
                        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].round().astype(int)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'DeterminationDateTime', 'Quantity']]
                    df['DeterminationDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['DeterminationDateTime'], unit="minutes")
                elif concept == 'BodyTemperature':
                    if table == "nurseCharting.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'nursingchartentryoffset':'MeasurementDateTime', 
                                                          'nursingchartvalue': 'Quantity'})
                        df = df[(df['nursingchartcelltypevallabel'] == 'Temperature') & (df['nursingchartcelltypevalname'] == 'Temperature (C)')]
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float)
                    elif table == "vitalPeriodic.csv.gz":
                        # checked
                        df = chunk.copy().rename(columns={'patientunitstayid':'SubjectPseudoIdentifier', 
                                                          'observationoffset': 'MeasurementDateTime', 
                                                          'temperature':'Quantity'})
                        df = df[df['Quantity'].notna()]
                        df['Quantity'] = df['Quantity'].astype(float)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                    df['MeasurementDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['MeasurementDateTime'], unit="minutes")
                       
                else:
                    raise Exception(f'Concept {concept} not supported')
                


                
                
                df.drop_duplicates(inplace=True)
                # add chunk to the correct concept file
                if not os.path.exists(path+concept+".parquet"):
                    df.to_parquet(path+concept+".parquet", engine='fastparquet')
                else:
                    df.to_parquet(path+concept+".parquet", engine='fastparquet', append=True) 

def load_mimic_concepts(table_to_concept_dict: dict):
    # delete all previous loaded concepts to overwrite
    path = 'data/concepts/mimic/'
    if not os.path.exists(path):
        os.makedirs(path)

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    # we handle each table seperately
    for table in table_to_concept_dict:
        print(table)

        # get all concepts associated to the specific table
        concepts = table_to_concept_dict[table]['concepts']
        print(concepts)

        # get the columns and their data types which we are interested in this table
        columns = list(table_to_concept_dict[table]['columns'].keys())
        # get date columns
        dates = [e for e in table_to_concept_dict[table]['columns'].keys() if table_to_concept_dict[table]['columns'][e] == 'datetime']
        # get non date columns
        dtype_without_dates = {key: value for key, value in table_to_concept_dict[table]['columns'].items() if key not in dates}


        #if table != 'hosp/labevents.csv.gz':
        #   continue

        # load the table in chunks
        df_chunks = pd.read_csv('data/raw/mimic/'+table, usecols=columns, parse_dates=dates, dtype=dtype_without_dates, chunksize=1000000)
        
        # we divide the loaded table into chunks
        for chunk in df_chunks:
            # for each concept which this table has information for, we extract the relevant data and add it to the concept file
            for concept in concepts:
                if concept == 'AdministrativeCase':
                    if table == 'hosp/admissions.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                        'admittime':'AdmissionDateTime',
                                                        'dischtime':'DischargeDateTime'})
                        df['Location'] = 'hosp'
                    elif table == 'icu/icustays.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                        'intime':'AdmissionDateTime',
                                                        'outtime':'DischargeDateTime'})
                        df['Location'] = 'icu'  
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','AdmissionDateTime','DischargeDateTime','Location']]
                elif concept == 'AdministrativeGender':
                    if table == 'hosp/patients.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                            'gender':'Code'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Code']]
                elif concept == 'DeathDate':
                    if table == 'hosp/admissions.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                            'deathtime':'DeathDateTime'})
                    elif table == 'hosp/patients.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                            'dod':'DeathDateTime'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','DeathDateTime']]
                elif concept == 'Ethnicity':
                    if table == 'hosp/admissions.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier',
                                                            'race':'Ethnicity'})
                        # take the first recorded ethnicity
                        df = df.sort_values(by='admittime')
                        df = df.groupby('SubjectPseudoIdentifier').first().reset_index()

                        df = df[(~df['Ethnicity'].isin(['OTHER', 'UNKNOWN'])) & df['Ethnicity'].notna()]

                    df = df[['SubjectPseudoIdentifier','Ethnicity']]
                elif concept == 'BirthDate':
                    if table == 'hosp/patients.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier'})
                        df['BirthDateTime'] = df['anchor_year'] - df['anchor_age']
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','BirthDateTime']]
                elif concept == 'LabResult':
                    if table == 'hosp/microbiologyevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 
                                                            'spec_type_desc':'LabTest', 
                                                            'org_name':'QualitativeResult',
                                                            'charttime': 'CollectionDateTime'})
                        df['ReportDateTime'] = pd.to_datetime(float('NaN'))
                        df['QuantitativeResult'] = float('NaN')
                        df = df[(df['LabTest'] == 'BLOOD CULTURE') & (df['QualitativeResult'] != 'CANCELLED')]
                    elif table == 'hosp/labevents.csv.gz' :
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 
                                                            'itemid':'LabTest', 
                                                            'value':'QualitativeResult', 
                                                            'valuenum':'QuantitativeResult',
                                                            'charttime': 'CollectionDateTime',
                                                            'storetime': 'ReportDateTime'})
                        # unfortunately we can't handle all labs at once so we have to make a selection
                        labs = []
                        from constant_files.lab_list import lab_list
                        for key in lab_list:
                            labs = labs + lab_list[key]

                        df = df[df['LabTest'].isin(labs)].copy()


                        # instead of the itemid, we use the name of the lab test
                        def find_key(dictionary, integer):
                            for key, value_list in dictionary.items():
                                if integer in value_list:
                                    return key
                            return None
                        
                        df['LabTest'] = df['LabTest'].astype(str)
                        for itemid in labs:
                            lab_test_name = find_key(lab_list, itemid)
                            df.loc[df['LabTest'] == str(itemid), 'LabTest'] = lab_test_name
                        df = df[df['QuantitativeResult'].notna()]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'LabTest', 'CollectionDateTime', 'ReportDateTime', 'QualitativeResult', 'QuantitativeResult']]
                elif concept == 'DrugAdministrationEvent':
                    if table == 'icu/inputevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'itemid': 'Drug', 'starttime':'StartDateTime'})
                        df = df[df['amount'] > 0]
                        df['Drug'] = df['Drug'].astype(str)

                        icu_drug = pd.read_csv('constant_files/icu_antibiotics.csv')['itemid']
                        df = df[df['Drug'].isin(icu_drug)]
                    elif table == 'hosp/emar.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier','medication': 'Drug', 'charttime':'StartDateTime'})
                        administerd_list = ['Administered', 'Partial Administered', 'Applied','Started', 'Delayed Administered','Restarted',
                                            'Rate Change', ' in Other Location', 'Administered in Other Location','Started in Other Location', 
                                            'Administered Bolus from IV Drip', 'Not Stopped','Applied in Other Location','Removed Existing / Applied New', 
                                            'Delayed Started', 'Delayed Applied','Not Stopped per Sliding Scale','Partial ', 'Removed Existing / Applied New in Other Location']
                        df = df[df['event_txt'].isin(administerd_list)]
                        emar_drug = pd.read_csv('constant_files/hosp_antibiotics_equal_to_icu.csv')['medication']
                        df = df[df['Drug'].isin(emar_drug)].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Drug','StartDateTime']]
                elif concept == 'BodyTemperature':
                    if table == 'icu/chartevents.csv.gz':
                        # 223762
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'MeasurementDateTime', 'valuenum': 'Quantity'})
                        df = df[df['itemid'] == 223762]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                elif concept == 'RespiratoryRate':
                    if table == 'icu/chartevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'DeterminationDateTime', 'valuenum': 'Quantity'})
                        df = df[df['itemid'] == 220210].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'DeterminationDateTime', 'Quantity']]
                elif concept == 'OxygenSaturation':
                    if table == 'icu/chartevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'MeasurementDateTime', 'valuenum': 'Quantity'})
                        df = df[df['itemid'] == 220277].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                elif concept == 'BloodPressure':
                    if table == 'icu/chartevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'MeasurementDateTime', 'valuenum': 'Quantity'})
                        systolic_list = [220050, 220179, 224167, 227243]
                        df_sys = df[df['itemid'].isin(systolic_list)].copy()
                        df_sys['Type'] = 'systolic'
                        diastolic_list = [220051, 220180, 224643, 227242]
                        df_dia = df[df['itemid'].isin(diastolic_list)].copy()
                        df_dia['Type'] = 'diastolic'
                        df = pd.concat([df_sys, df_dia])
                    elif table == 'hosp/omr.csv.gz':
                        df = chunk[chunk['result_name'] == 'Blood Pressure'].copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'chartdate':'MeasurementDateTime'})
                        df[['systolic', 'diastolic']] = df['result_value'].str.split('/', expand=True)
                        df[['systolic', 'diastolic']] = df[['systolic', 'diastolic']].astype(float)
                        # assume as late as possible 23.59 and take the largest seq number
                        df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'].dt.date.astype(str) + ' 23:59')
                        max_seq_num_indices = df.groupby(['SubjectPseudoIdentifier', 'MeasurementDateTime'])['seq_num'].idxmax()
                        df = df.loc[max_seq_num_indices, :]
                        df = df.melt(id_vars=['SubjectPseudoIdentifier', 'MeasurementDateTime'], 
                                            value_vars=['systolic', 'diastolic'],
                                            var_name='Type',
                                            value_name='Quantity')
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Type','Quantity']]
                # elif concept == 'SimpleScore': # ignore the defined scores they are almost empty
                # ignore them, the mimic tables are almost empty
                    # scores_list = {
                    #     'GCS - Eye Opening' : [1630825], # text
                    #     'GCS - Verbal Response' : [1627786], # text
                    #     'GCS - Motor Response' : [1623795], # text
                    #     'SOFA Score': [227428] # numeric
                    # }
                #      if table == 'icu/chartevents.csv.gz':
                #          df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'AssessmentDateTime', 'valuenum': 'Value', 'itemid':'ScoringSystem'})
                #          scores = []
                #          from constant_files.scores_list import scores_list
                #          for key in lab_list:
                #              scores = scores + scores_list[key]
                #          df = df[df['ScoringSystem'].isin(scores)].copy()
                #      else:
                #          raise Exception(f"Table {table} not supported")
                #      df = df[['SubjectPseudoIdentifier','AssessmentDateTime','ScoringSystem','Value']]
                elif concept == 'BodyWeight':
                    if table == 'hosp/omr.csv.gz':
                        df = chunk[chunk['result_name'] == 'Weight (Lbs)'].copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'chartdate':'MeasurementDateTime', 'result_value':'Quantity'})
                        df[['Quantity']] = df[['Quantity']].astype(float)
                        df['Quantity'] = df['Quantity']* 0.45359237 # convert to kg
                        df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'].dt.date.astype(str) + ' 23:59')
                        # to get most recent measurement
                        max_seq_num_indices = df.groupby(['SubjectPseudoIdentifier', 'MeasurementDateTime'])['seq_num'].idxmax()
                        df = df.loc[max_seq_num_indices, :]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                elif concept == 'BodyMassIndex':
                    if table == 'hosp/omr.csv.gz':
                        df = chunk[chunk['result_name'] == 'BMI (kg/m2)'].copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'chartdate':'DeterminationDateTime', 'result_value':'Quantity'})
                        df[['Quantity']] = df[['Quantity']].astype(float)
                        df['DeterminationDateTime'] = pd.to_datetime(df['DeterminationDateTime'].dt.date.astype(str) + ' 23:59')
                        max_seq_num_indices = df.groupby(['SubjectPseudoIdentifier', 'DeterminationDateTime'])['seq_num'].idxmax()
                        df = df.loc[max_seq_num_indices, :]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','DeterminationDateTime','Quantity']]
                elif concept == 'BodyHeight':
                    if table == 'hosp/omr.csv.gz':
                        df = chunk[chunk['result_name'] == 'Height (Inches)'].copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'chartdate':'MeasurementDateTime', 'result_value':'Quantity'})
                        df[['Quantity']] = df[['Quantity']].astype(float)
                        df['Quantity'] = df['Quantity']* 2.54 # convert to cm
                        df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'].dt.date.astype(str) + ' 23:59')
                        max_seq_num_indices = df.groupby(['SubjectPseudoIdentifier', 'MeasurementDateTime'])['seq_num'].idxmax()
                        df = df.loc[max_seq_num_indices, :]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                elif concept == 'HeartRate':
                    if table == 'icu/chartevents.csv.gz':
                        df = chunk.copy().rename(columns={'subject_id':'SubjectPseudoIdentifier', 'charttime':'MeasurementDateTime', 'valuenum': 'Quantity'})
                        df = df[df['itemid'] == 220045]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]            
                else:
                    raise Exception(f'Concept {concept} not supported')
                
                df.drop_duplicates(inplace=True)
                # add chunk to the correct concept file
                if not os.path.exists(path+concept+".parquet"):
                    df.to_parquet(path+concept+".parquet", engine='fastparquet')
                else:
                    df.to_parquet(path+concept+".parquet", engine='fastparquet', append=True)    


def load_pic_concepts(table_to_concept_dict: dict):
    # delete all previous loaded concepts to overwrite
    path = 'data/concepts/pic/'
    if not os.path.exists(path):
        os.makedirs(path)

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

    # we handle each table seperately
    for table in table_to_concept_dict:
        print(table)

        # get all concepts associated to the specific table
        concepts = table_to_concept_dict[table]['concepts']
        print(concepts)

        # get the columns and their data types which we are interested in this table
        columns = list(table_to_concept_dict[table]['columns'].keys())
        # get date columns
        dates = [e for e in table_to_concept_dict[table]['columns'].keys() if table_to_concept_dict[table]['columns'][e] == 'datetime']
        # get non date columns
        dtype_without_dates = {key: value for key, value in table_to_concept_dict[table]['columns'].items() if key not in dates}


        #if table != 'CHARTEVENTS.csv.gz':
        #   continue

        # load the table in chunks
        df_chunks = pd.read_csv('data/raw/pic/'+table, usecols=columns, parse_dates=dates, dtype=dtype_without_dates, chunksize=1000000)
        
        # we divide the loaded table into chunks
        for chunk in df_chunks:
            # for each concept which this table has information for, we extract the relevant data and add it to the concept file
            for concept in concepts:
                if concept == 'AdministrativeCase':
                    if table == 'ADMISSIONS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                        'ADMITTIME':'AdmissionDateTime',
                                                        'DISCHTIME':'DischargeDateTime'})
                        df['Location'] = 'hosp'
                    elif table == 'ICUSTAYS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                        'INTIME':'AdmissionDateTime',
                                                        'OUTTIME':'DischargeDateTime'})
                        df['Location'] = 'icu'  
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','AdmissionDateTime','DischargeDateTime','Location']]
                elif concept == 'AdministrativeGender':
                    if table == 'PATIENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                          'GENDER':'Code'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Code']]
                elif concept == 'DeathDate':
                    if table == 'ADMISSIONS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                            'DEATHTIME':'DeathDateTime'})
                    elif table == 'PATIENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                          'DOD':'DeathDateTime'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','DeathDateTime']]
                elif concept == 'Ethnicity':
                    if table == 'ADMISSIONS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                          'ETHNICITY':'Ethnicity'})
                        # take the first recorded ethnicity
                        df = df.sort_values(by='ADMITTIME')
                        df = df.groupby('SubjectPseudoIdentifier').first().reset_index()

                        df = df[(~df['Ethnicity'].isin(['OTHER', 'UNKNOWN'])) & df['Ethnicity'].notna()]

                    df = df[['SubjectPseudoIdentifier','Ethnicity']]
                elif concept == 'BirthDate':
                    if table == 'PATIENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                          'DOB':'BirthDateTime'})
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','BirthDateTime']]
                elif concept == 'LabResult':
                    if table == 'MICROBIOLOGYEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'SPEC_ITEMID':'LabTest', 
                                                          'ORG_ITEMID':'QualitativeResult',
                                                          'CHARTTIME': 'CollectionDateTime'})
                        df['ReportDateTime'] = pd.to_datetime(float('NaN'))
                        df['QuantitativeResult'] = float('NaN')
                        df = df[df['LabTest'].str.contains('LIS0162')] #| micro['SPEC_ITEMID'].str.contains('LIS05088') | micro['SPEC_ITEMID'].str.contains('LIS0156')| micro['SPEC_ITEMID'].str.contains('LIS0567')
                    elif table == 'LABEVENTS.csv.gz' :
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'ITEMID':'LabTest', 
                                                          'VALUE':'QualitativeResult', 
                                                          'VALUENUM':'QuantitativeResult',
                                                          'CHARTTIME': 'CollectionDateTime'})
                        # unfortunately we can't handle all labs at once so we have to make a selection
                        df['ReportDateTime'] = df['CollectionDateTime']
                        labs = []
                        from constant_files.lab_list import lab_list_pic
                        for key in lab_list_pic:
                            labs = labs + lab_list_pic[key]

                        df = df[df['LabTest'].isin(labs)].copy()


                        # instead of the itemid, we use the name of the lab test
                        def find_key(dictionary, integer):
                            for key, value_list in dictionary.items():
                                if integer in value_list:
                                    return key
                            return None
                        
                        df['LabTest'] = df['LabTest'].astype(str)
                        for itemid in labs:
                            lab_test_name = find_key(lab_list_pic, itemid)
                            df.loc[df['LabTest'] == str(itemid), 'LabTest'] = lab_test_name
                        df = df[df['QuantitativeResult'].notna()]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'LabTest', 'CollectionDateTime', 'ReportDateTime', 'QualitativeResult', 'QuantitativeResult']]
                elif concept == 'DrugAdministrationEvent':
                    if table == 'PRESCRIPTIONS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'DRUG_NAME_EN': 'Drug', 
                                                          'STARTDATE':'StartDateTime',
                                                          'ENDDATE':'EndDateTime'})

                        #df['StartDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['StartDateTime'], unit="minutes")
                        #df['EndDateTime'] = pd.to_datetime('2000-01-01 00:00') +  pd.to_timedelta(df['EndDateTime'], unit="minutes")

                        df = df[df['StartDateTime'] <= df['EndDateTime']]

                        df = df[~df['Drug'].isna()]

                        drugs = pd.read_csv("constant_files/pic_prescriptions.csv")     

                        df = df[(df['Drug'].isin(list(drugs['DRUG_NAME_EN'])))].copy()
                        

                        # we repeat the medication fact every day
                        #display(df.head(50))
                        dfs = []
                        
                        for _, row in df.iterrows():
                            admission_date = pd.to_datetime(row['StartDateTime'])
                            discharge_date = pd.to_datetime(row['EndDateTime'])
                            days = (discharge_date - admission_date).days + 1

                            # start and end
                            subject_df = pd.DataFrame({
                                'SubjectPseudoIdentifier': [row['SubjectPseudoIdentifier']] * 2,
                                'Drug': [row['Drug']] * 2,
                                'StartDateTime': [row['StartDateTime'], row['EndDateTime']]   
                            })
                            dfs.append(subject_df)

                            # in between
                            r = pd.date_range(start=admission_date, end=discharge_date, inclusive='neither', freq='D')
                            if len(r) != 0:
                                subject_df = pd.DataFrame({
                                    'SubjectPseudoIdentifier': [row['SubjectPseudoIdentifier']] * len(r),
                                    'Drug': [row['Drug']] * len(r),
                                    'StartDateTime': r
                                })
                                dfs.append(subject_df)
                        df = pd.concat(dfs, ignore_index=True)
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','Drug','StartDateTime']]
                elif concept == 'BodyTemperature':
                    if table == 'CHARTEVENTS.csv.gz':
                        # 223762
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1001]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                elif concept == 'RespiratoryRate':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'DeterminationDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1004].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'DeterminationDateTime', 'Quantity']]
                elif concept == 'OxygenSaturation':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1006].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]
                elif concept == 'BloodPressure':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier',
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        systolic_list = [1016]
                        df_sys = df[df['ITEMID'].isin(systolic_list)].copy()
                        df_sys['Type'] = 'systolic'
                        diastolic_list = [1015]
                        df_dia = df[df['ITEMID'].isin(diastolic_list)].copy()
                        df_dia['Type'] = 'diastolic'
                        df = pd.concat([df_sys, df_dia])
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Type','Quantity']]
                elif concept == 'BodyWeight':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1014].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                elif concept == 'BodyHeight':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1013].copy()
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier','MeasurementDateTime','Quantity']]
                elif concept == 'HeartRate':
                    if table == 'CHARTEVENTS.csv.gz':
                        df = chunk.copy().rename(columns={'SUBJECT_ID':'SubjectPseudoIdentifier', 
                                                          'CHARTTIME':'MeasurementDateTime', 
                                                          'VALUENUM': 'Quantity'})
                        df = df[df['ITEMID'] == 1003]
                    else:
                        raise Exception(f"Table {table} not supported")
                    df = df[['SubjectPseudoIdentifier', 'MeasurementDateTime', 'Quantity']]            
                else:
                    raise Exception(f'Concept {concept} not supported')
                
                df.drop_duplicates(inplace=True)
                # add chunk to the correct concept file
                if not os.path.exists(path+concept+".parquet"):
                    df.to_parquet(path+concept+".parquet", engine='fastparquet')
                else:
                    df.to_parquet(path+concept+".parquet", engine='fastparquet', append=True)   
