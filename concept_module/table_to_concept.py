# this is a list to harmonize the data view between multiple data sources
# mimic iv 2.2
# eicu 2.0 (for simplicitly forget the linkage with the uniquepid)
# pic 1.1.0

# Table to Concept mapping

mimic_table_to_concept = {
    "hosp/admissions.csv.gz": {
        "columns":{"subject_id":int, 
                   "admittime": "datetime", 
                   "dischtime": "datetime",
                   "deathtime": "datetime", 
                   "race": str}, 
        "concepts":["AdministrativeCase", "DeathDate", "Ethnicity"]
    },
    "icu/icustays.csv.gz": {
        "columns":{"subject_id":int, 
                   "intime": "datetime", 
                   "outtime": "datetime"}, 
        "concepts":["AdministrativeCase"]
    },
    "hosp/patients.csv.gz" : {
        "columns":{"subject_id":int, 
                   "anchor_year": int, 
                   "anchor_age": int, 
                   "dod": "datetime",
                   "gender": str},
        "concepts":["BirthDate", "DeathDate", "AdministrativeGender"]
    },
    "icu/chartevents.csv.gz" : { # 7min 30s
        "columns":{"subject_id":int, 
                   "charttime": "datetime", 
                   "itemid": int,
                   "value":str, 
                   "valuenum":float, 
                   "valueuom":str},
        "concepts" : ["BodyTemperature", "BloodPressure", "HeartRate", "RespiratoryRate", "OxygenSaturation"] #, "SimpleScore"
    },
    "hosp/labevents.csv.gz" : { #5min 30s
        "columns":{"subject_id":int, 
                   "charttime": "datetime", 
                   "storetime": "datetime", 
                   "itemid":int, 
                   "value":str, 
                   "valuenum":float},
        "concepts" : ["LabResult"]
        
    },
    "hosp/omr.csv.gz" : { # 7min 45s
        "columns":{"subject_id":int, 
                   "chartdate": "datetime", 
                   "seq_num": int,
                   "result_name":str, 
                   "result_value":str},
        "concepts" : ["BodyWeight", "BodyMassIndex", "BodyHeight", "BloodPressure"]
    },
    "hosp/emar.csv.gz" : {
        "columns":{"subject_id":int, 
                   "charttime": "datetime", 
                   "medication":str, 
                   "event_txt":str},
        "concepts" : ["DrugAdministrationEvent"]
        # the same named J01 antibiotics from icu_antibiotics.csv but with their corresponding emar name
    },
    "icu/inputevents.csv.gz" : {
        "columns":{"subject_id":int, 
                   "starttime": "datetime", 
                   "itemid":int, 
                   "amount":float},
        "concepts" : ["DrugAdministrationEvent"]
    },
    "hosp/microbiologyevents.csv.gz" : {
        "columns":{"subject_id":int, 
                   "charttime": "datetime", 
                   "spec_type_desc":str, 
                   "org_name":str},
        "concepts" : ["LabResult"]
        # spec_type_desc=BLOOD CULTURE
        # spec_type_desc!=BLOOD CULTURE
    }
}

pic_table_to_concept = {
    "ADMISSIONS.csv.gz": {
        "columns":{"SUBJECT_ID":int, 
                   "ADMITTIME": "datetime", 
                   "DISCHTIME": "datetime",
                   "DEATHTIME": "datetime", 
                   "ETHNICITY": str}, 
        "concepts":["AdministrativeCase", "DeathDate", "Ethnicity"]
    },
    "ICUSTAYS.csv.gz": {
        "columns":{"SUBJECT_ID":int, 
                   "INTIME": "datetime", 
                   "OUTTIME": "datetime"}, 
        "concepts":["AdministrativeCase"]
    },
    "PATIENTS.csv.gz" : {
        "columns":{"SUBJECT_ID":int, 
                   "DOB": "datetime", 
                   "DOD": "datetime",
                   "GENDER": str},
        "concepts":["BirthDate", "DeathDate", "AdministrativeGender"]
    },
    "CHARTEVENTS.csv.gz" : { # 7min 30s
        "columns":{"SUBJECT_ID":int, 
                   "CHARTTIME": "datetime", 
                   "ITEMID": int,
                   "VALUE":str, 
                   "VALUENUM":float, 
                   "VALUEUOM":str},
        "concepts" : ["BodyTemperature", "BloodPressure", "HeartRate", "RespiratoryRate", "OxygenSaturation", "BodyWeight", "BodyHeight"] #, "SimpleScore"
    },
    "LABEVENTS.csv.gz" : { #5min 30s
        "columns":{"SUBJECT_ID":int, 
                   "CHARTTIME": "datetime", 
                   "ITEMID":int, 
                   "VALUE":str, 
                   "VALUENUM":float},
        "concepts" : ["LabResult"]
        
    },
    "PRESCRIPTIONS.csv.gz" : {
        "columns":{"SUBJECT_ID":int, 
                   "STARTDATE": "datetime", 
                   "ENDDATE": "datetime", 
                   "DRUG_NAME_EN":str},
        "concepts" : ["DrugAdministrationEvent"]
        # the same named J01 antibiotics from icu_antibiotics.csv but with their corresponding emar name
    },
    "MICROBIOLOGYEVENTS.csv.gz" : {
        "columns":{"SUBJECT_ID":int, 
                   "CHARTTIME": "datetime", 
                   "SPEC_ITEMID":str, 
                   "ORG_ITEMID":str},
        "concepts" : ["LabResult"]
    }
}



eicu_table_to_concept = {
    # "admissiondrug.csv.gz" : {
    #     "columns":{"patientunitstayid":int, "drugoffset": int, "drugname": str, "drugdosage": float},
    #     "concepts" : ["DrugAdministrationEvent"]
    # }
    "infusionDrug.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "infusionoffset": int, 
                   "drugname": str, 
                   "drugrate": str, 
                   "infusionrate": str,
                   "drugamount": str,
                   "volumeoffluid": str},
        "concepts" : ["DrugAdministrationEvent"]
    },
    "lab.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "labresultoffset": int, 
                   "labresultrevisedoffset": int,
                   "labname": str, 
                   "labresult": float, 
                   "labresulttext": str},
        "concepts" : ["LabResult"]
    },
    "medication.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "drugstartoffset": int, 
                   "drugordercancelled": str, 
                   "drugname": str, 
                   "drughiclseqno": float, 
                   "drugstopoffset": int,
                   "gtc": float},
        "concepts" : ["DrugAdministrationEvent"]
    },
    "microLab.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "culturetakenoffset": int, 
                   "culturesite": str, 
                   "organism": str},
        "concepts" : ["LabResult"]
    },
    "patient.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "patienthealthsystemstayid": int, 
                   "hospitaladmitoffset": int, 
                   "hospitaldischargeoffset": int, 
                   "unitdischargeoffset": int, 
                   "age": str, 
                   "admissionweight": float,
                   "admissionheight": float,
                   "unitdischargestatus": str, 
                   "hospitaldischargestatus": str, 
                   "gender": str, 
                   "ethnicity": str},
        "concepts":["AdministrativeCase", "AdministrativeGender", "BirthDate", "BodyHeight", "BodyMassIndex", "BodyWeight", "DeathDate", "Ethnicity"]
    },
    "nurseCharting.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "nursingchartentryoffset":int,
                   "nursingchartcelltypevalname": str,
                   "nursingchartcelltypevallabel": str,
                   "nursingchartvalue": str
                   },
        "concepts":["BloodPressure", "BodyTemperature", "HeartRate", "OxygenSaturation", "RespiratoryRate"]
    },
    "vitalPeriodic.csv.gz" : {
        "columns":{"patientunitstayid":int, 
                   "observationoffset":int,
                   "temperature": float, 
                   "heartrate": float, 
                   "respiration": float, 
                   "systemicsystolic": float, 
                   "systemicdiastolic": float, 
                   "sao2": float},
        "concepts":["BloodPressure", "BodyTemperature", "HeartRate", "OxygenSaturation", "RespiratoryRate"]
    }
}
