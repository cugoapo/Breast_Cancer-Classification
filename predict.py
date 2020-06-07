import preprocess as pf
import config


def predict(data):

    # Drop Unnecessary Column

    data = pf.drop_unnecessary_features(data, config.DROP_COLUMN_NAME)

    # Power Transformation

    data = pf.transform_features(data, config.POWER_TRANSFORM_VAR, config.TRANSFORMER_PATH)

    # scale
    data = pf.scale_features(data[config.FEATURES], config.SCALER_PATH )

    # NCA Features

    data = pf.nca_features(data, config.NCA_PATH)

    # Prediction

    prediction = pf.predict(data, config.MODEL_PATH)
    prediction_proba = pf.predict_proba(data, config.MODEL_PATH)
    return prediction, prediction_proba

