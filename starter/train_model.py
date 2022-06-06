import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from .ml.model import train_model
from .ml.data import process_data


def split_data(path):
    """
    Loads data in path and returns train and test splits

    Parameters
    ----------
    path: str
        Path to cleaned data(.scv)

    Returns
    -------
    df_train , df_test
    """
    data = pd.read_csv(f"{path}/data/cleaned/census.csv")
    df_train, df_test = train_test_split(data, test_size=0.20)
    return df_train, df_test


def train_persist_model(path, train_data):
    """
    Train model on training data and persist it

    Parameters
    ----------
    path: str
        Path to the root
    train_data: pd.DataFrame
        Dataframe to perform model training

    Returns
    -------
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    x_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features,
        label="salary", training=True
    )
    # train model
    model = train_model(x_train, y_train)
    # persist model and artifact (encoders, etc)
    dump(model, f"{path}/model/model.joblib")
    dump(encoder, f"{path}/model/encoder.joblib")
    dump(lb, f"{path}/model/lb.joblib")
