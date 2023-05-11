import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


train_path = 'data/train'
test_path = 'data/test'
LABEL_COL = 'SepsisLabel'


def process_file(file_path, avg_df, fill=False):
    df = pd.read_csv(file_path, sep='|')
    """ df = df.drop(columns=['HR','EtCO2','BaseExcess','HCO3','FiO2' ,'pH','PaCO2','SaO2','AST','BUN','Alkalinephos',
                          'Calcium','Chloride','Creatinine','Bilirubin_direct','Lactate','Magnesium','Phosphate',
                          'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen',
                          'Platelets'])"""

    if 1 in df[LABEL_COL].values:
        idx = df.index[df[LABEL_COL] == 1][0]
    else:
        idx = -1

    if idx >= 0:
        new_df = df.loc[:idx]
        y = 1
    else:
        new_df = df
        y = 0
    new_df = new_df[new_df.columns.drop([LABEL_COL])]
    if fill:
        X = new_df.mean().fillna(avg_df)
    else:
        X = new_df
    return X, y


def process_file_last(file_path, avg_df, fill=False):
    df = pd.read_csv(file_path, sep='|')
    """ df = df.drop(columns=['HR','EtCO2','BaseExcess','HCO3','FiO2' ,'pH','PaCO2','SaO2','AST','BUN','Alkalinephos',
                          'Calcium','Chloride','Creatinine','Bilirubin_direct','Lactate','Magnesium','Phosphate',
                          'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen',
                          'Platelets'])"""

    if 1 in df[LABEL_COL].values:
        idx = df.index[df[LABEL_COL] == 1][0]
    else:
        idx = -1

    if idx >= 0:
        new_df = df.loc[:idx]
        y = 1
    else:
        new_df = df
        y = 0
    new_df = new_df[new_df.columns.drop([LABEL_COL])]
    if fill:
        new_df = new_df.ffill().fillna(new_df.mean()).fillna(avg_df)
    df = new_df
    last_hour_data = df.tail(1).reset_index(drop=True)
    avg_data = df.mean().to_frame().T.reset_index(drop=True)
    first_hour_data = df.iloc[0]
    diff_data = last_hour_data.subtract(first_hour_data).reset_index(drop=True)
    result = pd.concat([last_hour_data, avg_data, diff_data], axis=1)
    result.columns = ['last_hour_' + str(col) for col in result.columns[:40]] + ['avg_' + str(col) for col in
                            result.columns[40:80]] + ['diff_' + str(col) for col in result.columns[80:]]
    X = result
    return X, y


def process_set(path, avg_df, fill=False, last=False):
    feature_list = []
    label_list = []
    name_list = []
    i = 0
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if last:
            X, y = process_file_last(file_path, avg_df, fill)
        else:
            X, y = process_file(file_path, avg_df, fill)
        feature_list.append(X)
        label_list.append(y)
        name_list.append(file_name)
        """i += 1
        if i > 80:
            break"""
    if fill:
        df = pd.concat(feature_list)
        df['label'] = label_list
        df['patient'] = name_list
        df.set_index('patient', inplace=True)
        return df
    return feature_list


if __name__ == '__main__':
    train_avgs = process_set(train_path, 0, False)
    avg_df = pd.concat(train_avgs).mean()
    #avg_df.to_pickle('avgs.pkl')
    train_df = process_set(train_path, avg_df, True, True)
    test_df = process_set(test_path, avg_df, True, True)
    X_train, y_train = train_df.drop(columns=['label']), train_df['label']
    X_test, y_test = test_df.drop(columns=['label']), test_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    """
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001]
    }
    f1_scores = []
    for est in [100, 200, 300]:
        for depth in [3, 5, 7]:
            for lr in [0.1, 0.01, 0.001]:
                xgb_model = xgb.XGBClassifier(n_estimators=est, max_depth=depth, learning_rate=lr)
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_val)
                f1 = f1_score(y_val, y_pred)
                f1_scores.append(f1)
                print(f"n_estimators: {est}, depth: {depth}, lr: {lr}")
                print("{} F1-score on validation set: {:.2f}".format('XGB', f1))
    """
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4]
    }
    f1_scores = []
    for est in [100, 200, 300]:
        for depth in [5, 10]:
            for split in [2, 4, 8]:
                for leaf in [1, 2, 4]:
                    xgb_model = RandomForestClassifier(n_estimators=est, max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
                    xgb_model.fit(X_train, y_train)
                    y_pred = xgb_model.predict(X_val)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append(f1)
                    print(f"n_estimators: {est}, depth: {depth}, split: {split}, leaf: {leaf}")
                    print("{} F1-score on validation set: {:.2f}".format('XGB', f1))



