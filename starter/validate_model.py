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
    model = load(f"{root_dir}/model/model.joblib")
    encoder = load(f"{root_dir}/model/encoder.joblib")
    lb = load(f"{root_dir}/model/lb.joblib")

    validate_on_slices(
        model,
        encoder,
        cat_features,
        lb,
        path,
        test_df)
