from joblib import load
from .ml.model import validate_on_slices


def val_model(test_df, cat_features, path):
    """

    Parameters
    ----------
    test_df
    cat_features
    path

    Returns
    -------

    """
    # load model and encoder
    model = load(f"{path}/model/model.joblib")
    encoder = load(f"{path}/model/encoder.joblib")
    lb = load(f"{path}/model/lb.joblib")

    validate_on_slices(
        model,
        encoder,
        cat_features,
        lb,
        path,
        test_df)
