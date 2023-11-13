import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import re

def get_res_df(df:pd.DataFrame):
    df = pd.DataFrame({
                            'precision':[precision_score(df['lot<5d'], df['pred'])],
                            'recall':[recall_score(df['lot<5d'], df['pred'])],
                            'f1':[f1_score(df['lot<5d'], df['pred'])],
                            'balanced_accuracy':[balanced_accuracy_score(df['lot<5d'], df['pred'])], # takes predictions
                            'prc_auc':[average_precision_score(df['lot<5d'], df['True'])], # takes probabilites
                            'roc_auc':[roc_auc_score(df['lot<5d'], df['True'])]#, # takes probabilites
                            })
    return df

def get_standard_stats(gt, preds, preds_proba):
    df = pd.DataFrame({
            'precision':[precision_score(gt, preds)],
            'recall':[recall_score(gt, preds)],
            'f1':[f1_score(gt, preds)],
            'balanced_accuracy':[balanced_accuracy_score(gt, preds)], # takes predictions
            'prc_auc':[average_precision_score(gt, preds_proba)], # takes probabilites
            'roc_auc':[roc_auc_score(gt, preds_proba)], # takes probabilites
        })
    return df


def remove_special_characters(input_string):
        pattern = r'[^a-zA-Z0-9_]'
        result = re.sub(pattern, '', input_string)
        return result