PATH_DATASET = "Dataset/breast-cancer.csv"

DROP_COLUMN_NAME = "id"

MISSING_VALUE = ""

TARGET_COLUMN = "diagnosis"

FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
            'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
            'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
            'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']


OUTLIER_THRESHOLD = -2.5

POWER_TRANSFORM_VAR = "area_se"

TRANSFORMER_PATH = "knn_model/transformer.pkl"

SCALER_PATH = "knn_model/scaler.pkl"

NCA_PATH = "knn_model/nca.pkl"

MODEL_PATH = "knn_model/KNN.pkl"

KNN_RANGE = list(range(1, 31))

KNN_WEIGHT_POTIONS = ["uniform", "distance"]

CrossValidation_CV = 10

