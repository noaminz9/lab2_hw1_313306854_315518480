import sys
import pandas as pd
import os
import xgboost as xgb


LABEL_COL = 'SepsisLabel'


def process_file_last(file_path, avg_df):
    df = pd.read_csv(file_path, sep='|')
    if 1 in df[LABEL_COL].values:
        idx = df.index[df[LABEL_COL] == 1][0]
    else:
        idx = -1

    if idx >= 0:
        new_df = df.loc[:idx]
    else:
        new_df = df
    df = new_df[new_df.columns.drop([LABEL_COL])]
    num_cols = len(df.columns())
    df = df.ffill().fillna(avg_df)
    last_hour_data = df.tail(1).reset_index(drop=True)
    avg_data = df.mean().to_frame().T.reset_index(drop=True)
    first_hour_data = df.iloc[0]
    diff_data = last_hour_data.subtract(first_hour_data).reset_index(drop=True)
    result = pd.concat([last_hour_data, avg_data, diff_data], axis=1)
    result.columns = ['last_hour_' + str(col) for col in result.columns[:num_cols]] + ['avg_' + str(col) for col in
                    result.columns[num_cols:2*num_cols]] + ['diff_' + str(col) for col in result.columns[2*num_cols:]]
    X = result
    return X


def process_set(path, avg_df):
    feature_list = []
    name_list = []
    i = 0
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        X = process_file_last(file_path, avg_df)
        feature_list.append(X)
        name = file_name[:-4]
        name_list.append(name)
        """i += 1
        if i > 80:
            break"""

    df = pd.concat(feature_list)
    df['id'] = name_list
    df.set_index('id', inplace=True)
    return df


if __name__ == '__main__':
    avg_df = pd.read_pickle('avgs.pkl')
    path = sys.argv[1]
    test_df = process_set(path, avg_df)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.001)
    model.load_model("xgboost_model.bin")
    y_pred = model.predict(test_df)
    test_df['prediction'] = y_pred
    test_df.reset_index(inplace=True)
    test_df = test_df[['id', 'prediction']]
    test_df['ID_number'] = test_df['id'].str.extract('(\d+)').astype(int)
    df_sorted = test_df.sort_values('ID_number')
    df_sorted.drop('ID_number', axis=1, inplace=True)
    df_sorted.to_csv('prediction.csv', index=False)


