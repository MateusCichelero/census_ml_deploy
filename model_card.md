# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The training pipeline was based on the use of sciklearn RandomForestClassifier, an implementation of Random Forest algorithm, with n_estimators=100 as the only non-default hyperparameter. 

## Intended Use
This model is intented to be used with a person financial data as input with the main goal to predict its salary category (<=50k or >50k)

## Training Data
To train the model, it was used publicly available data from the Census Income Data Set https://archive.ics.uci.edu/ml/datasets/census+income. It was adopted a hold-out split strategy using 80% of data for training. 

## Evaluation Data
Using the same original dataset, with 20% hold-out split form model evaluation

## Metrics
Different evaluation metrics were applied: Precision,Recall, Accuracy score, F1 score. Accuracy for the 5 fold cv set were mean:0.825, std:0.006. For the other metrics regarding different slices of data, see the slice_output.txt file.

## Ethical Considerations
Evaluation on different slices of data were performed for further investigation on model fairness.

## Caveats and Recommendations
A superficial analysis on performance data already shows us that the model is biased considering gender (Male/Female). Also, some particular profiles are underrepresented, so it is important to continue improving in this case.
