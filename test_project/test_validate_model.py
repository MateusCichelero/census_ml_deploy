import os
from starter.validate_model import val_model


def test_model(clean_data, cat_features):
    val_model(clean_data, cat_features, path='./')

    assert os.path.isfile("./model/slice_output.txt")
