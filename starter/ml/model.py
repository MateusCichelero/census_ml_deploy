import logging
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Applying cv to log the mean and std accuracy scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, x):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    x : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(x)
    return preds


def validate_on_slices(model, encoder, categorical_features,
                       lb, path, test_data):
    """
    Compute validation metrics for each categorical element inside a
    categorical feature
    Parameters
    ----------
    model: RandomForestClassifier
        Trained machine learning model.
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    lb: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer
    path: str
        Path to the root
    test_data: np.array
        Data used for validation on slices.

    Returns
    -------

    """
    with open(f'{path}/model/slice_output.txt', 'w') as file:
        for category in categorical_features:
            for categorical_element in test_data[category].unique():
                temp_df = test_data[test_data[category] == categorical_element]

                x_test, y_test, _, _ = process_data(
                    temp_df,
                    categorical_features=categorical_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

                y_pred = model.predict(x_test)

                prc, rcl, fb = compute_model_metrics(y_test, y_pred)

                metric_info = "[%s]-[%s] Precision: %s " \
                              "Recall: %s FBeta: %s" % (category,
                                                        categorical_element,
                                                        prc, rcl, fb)
                logging.info(metric_info)
                file.write(metric_info + '\n')
