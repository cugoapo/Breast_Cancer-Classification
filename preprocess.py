import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
import joblib


# Individual pre-processing and training functions
# ================================================

# 1 Load Data
def load_data(df_path):
    return pd.read_csv(df_path)


# 2 Drop Unnecessary Columns
def drop_unnecessary_features(df, drop_col_path):
    return df.drop([drop_col_path], axis=1)


# 3 Outlier Detection and Drop Outliers
def outlier_detection(df, target_path, threshold_path):
    x = df.drop(target_path, axis=1)
    y = df[target_path]

    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(x)
    X_score = clf.negative_outlier_factor_

    outlier_score = pd.DataFrame()
    outlier_score["score"] = X_score

    filt = outlier_score["score"] < threshold_path
    outlier_index = outlier_score[filt].index.tolist()

    # drop outlier
    x = x.drop(outlier_index)
    y = y.drop(outlier_index)

    df = pd.concat([x, y], axis=1)
    return df


# 4 Power Transformation
def train_transformer(df, pow_trans_var_path, output_path):
    transformer = PowerTransformer(standardize=False, copy=False)
    transformer.fit(df[pow_trans_var_path].values.reshape(-1, 1))
    joblib.dump(transformer, output_path)
    transformer.transform(df[pow_trans_var_path].values.reshape(-1, 1))
    return df


def transform_features(df, pow_trans_var_path, transformer):
    transformer = joblib.load(transformer)
    transformer.transform(df[pow_trans_var_path].values.reshape(-1, 1))
    return df

# 5 Standardization
def train_scaler(df, output_path):
    scaler = RobustScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler


def scale_features(df, scaler):
    scaler = joblib.load(scaler)  # with joblib probably
    return scaler.transform(df)


# 6 NCA train and save
def train_nca(x_scaled, target, output_path):
    nca = NeighborhoodComponentsAnalysis(n_components=2, random_state=42)
    nca.fit(x_scaled, target)
    joblib.dump(nca, output_path)
    x_reduced_nca = nca.transform(x_scaled)
    return x_reduced_nca


# NCA Transform Features for prediction
def nca_features(df, nca):
    nca = joblib.load(nca)
    return nca.transform(df)


# 7 Divide Dataset
def divide_train_test(nca_data, df, target):
    X_train, X_test, y_train, y_test = train_test_split(nca_data,
                                                        df[target],
                                                        test_size=0.33,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


# 8 Training ,Tuning and Saving Model (best hyper parameter)
def train_knn(X_train, X_test, y_train, y_test, KNN_RANGE, KNN_WEIGHT_POTIONS,cv, output_path):
    k_range = KNN_RANGE
    weight_potions = KNN_WEIGHT_POTIONS
    print()
    param_grid = {'n_neighbors': k_range, 'weights': weight_potions}

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=cv, scoring="accuracy")
    grid.fit(X_train, y_train)

    print("Best Training Score: {} with parameters: {} ".format(grid.best_score_, grid.best_params_))
    print()

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(X_train, y_train)
    joblib.dump(knn, output_path)

    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)

    print("Test Score : {}, Train Score : {}".format(acc_test, acc_train))
    print()
    print("Conf Matrix Test\n", cm_test)
    print("Conf Matrix Train\n", cm_train)

    return None


# 9 Prediction
def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)


# 10 Prediction Probability
def predict_proba(df, model):
    model = joblib.load(model)
    return model.predict_proba(df)



