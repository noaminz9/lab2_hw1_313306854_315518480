import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
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
        new_df = new_df.ffill().fillna(avg_df)
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
            """if 1 in df['Gender'].values:
                X, y = process_file_last(file_path, avg_df, fill)
            else:
                i += 1
                continue"""

        else:
            X, y = process_file(file_path, avg_df, fill)
        feature_list.append(X)
        label_list.append(y)
        name_list.append(file_name)
    if fill:
        df = pd.concat(feature_list)
        df['label'] = label_list
        df['patient'] = name_list
        df.set_index('patient', inplace=True)
        return df
    return feature_list


if __name__ == '__main__':
    #train_avgs = process_set(train_path, 0, False)
    #avg_df = pd.concat(train_avgs).mean()
    #avg_df.to_pickle('avgs.pkl')
    avg_df = pd.read_pickle('avgs.pkl')
    train_df = process_set(train_path, avg_df, True, True)
    test_df = process_set(test_path, avg_df, True, True)
    X_train, y_train = train_df.drop(columns=['label']), train_df['label']
    X_test, y_test = test_df.drop(columns=['label']), test_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, min_samples_leaf=1),
        SVC(),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(max_iter=1000),
        xgb.XGBClassifier(n_estimators=100, max_depth=5)
    ]

    # Training and evaluating each model on the validation set
    f1_scores = []
    train_sizes, train_scores, val_scores = [], [], []
    for model in models:
        model_name = type(model).__name__
        print("Training model: {}".format(model_name))

        # Calculating learning curve for the model
        train_sizes_, train_scores_, val_scores_ = learning_curve(model, X_train, y_train, cv=5, scoring='f1',
                                                                  n_jobs=-1)
        train_sizes.append(train_sizes_)
        train_scores.append(train_scores_)
        val_scores.append(val_scores_)

        # Training the model on the full training set
        model.fit(X_train, y_train)

        # Evaluating the model on the validation set
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)
        print("{} F1-score on validation set: {:.2f}".format(model_name, f1))

    plt.figure(figsize=(15, 10))
    for i, model_name in enumerate([type(model).__name__ for model in models]):
        train_scores_mean = np.mean(train_scores[i], axis=1)
        train_scores_std = np.std(train_scores[i], axis=1)
        plt.plot(train_sizes[i], train_scores_mean, label="Train F1-score: " + model_name, marker='o')
        plt.fill_between(train_sizes[i], train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1)

    plt.title("Train F1-score comparison of different models")
    plt.xlabel("Number of training examples")
    plt.ylabel("F1-score")
    plt.legend(loc="best")
    plt.savefig('train_graph.png')
    plt.show()


    plt.figure(figsize=(15, 10))
    for i, model_name in enumerate([type(model).__name__ for model in models]):
        val_scores_mean = np.mean(val_scores[i], axis=1)
        val_scores_std = np.std(val_scores[i], axis=1)
        plt.plot(train_sizes[i], val_scores_mean, label="Validation F1-score: " + model_name, marker='o')
        plt.fill_between(train_sizes[i], val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1)

    plt.title("Validation F1-score comparison of different models")
    plt.xlabel("Number of training examples")
    plt.ylabel("F1-score")
    plt.legend(loc="best")
    plt.savefig('validation_graph.png')
    plt.show()



    best_model = models[f1_scores.index(max(f1_scores))]
    best_model.save_model("xgboost_model.bin")

    # Using the best model on the test set
    y_pred_test = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test)

    print("Best model F1-score on validation set: {:.2f}".format(max(f1_scores)))
    print("Best model F1-score on test set: {:.2f}".format(test_f1))

