import os
from starter.train_model import split_data, train_persist_model


def test_split_data():
    train_df, test_df = split_data(root_path='./')

    assert train_df.shape[0] > 0
    assert train_df.shape[1] == 12

    assert test_df.shape[0] > 0
    assert test_df.shape[1] == 12


def test_train_persist_model(clean_data, cat_features):

    train_persist_model('./', clean_data)

    assert os.path.isfile("./model/model.joblib")
    assert os.path.isfile("./model/encoder.joblib")
    assert os.path.isfile("./model/lb.joblib")