import numpy as np
import preprocess as pf
import config
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
import pandas as pd


# Load data
data = pf.load_data(config.PATH_DATASET)

# Drop Unnecessary Column

data = pf.drop_unnecessary_features(data, config.DROP_COLUMN_NAME)


# detect outliers and drop outliers

data = pf.outlier_detection(data, config.TARGET_COLUMN, config.OUTLIER_THRESHOLD)


# Power Transformation

data = pf.train_transformer(data, config.POWER_TRANSFORM_VAR, config.TRANSFORMER_PATH)


# train scaler and save
scaler = pf.train_scaler(data[config.FEATURES],
                         config.SCALER_PATH)

# scale train set
x_scaled = scaler.transform(data[config.FEATURES])


# train NCA Model

x_reduced_nca = pf.train_nca(x_scaled, data[config.TARGET_COLUMN], config.NCA_PATH)


# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(x_reduced_nca, data, config.TARGET_COLUMN)


# train model and save

pf.train_knn(X_train, X_test, y_train, y_test,
             config.KNN_RANGE, config.KNN_WEIGHT_POTIONS,
             config.CrossValidation_CV, config.MODEL_PATH)

print('Finished training')


