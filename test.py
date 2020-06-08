import config

import preprocess as pf
from predict import predict


def test_make_single_prediction():
    # Given
    test_data = pf.load_data(config.PATH_DATASET)
    single_test_json = test_data[0:1]

    # When
    subject = predict(single_test_json)[0]

    # Then
    assert subject is not None
    assert isinstance(subject[0], str)
    assert (subject[0]) == "M"
