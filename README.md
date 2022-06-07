# Census Salary Prediction ML Model on Heroku with FastAPI

- Project **Deploying a Machine Learning Model on Heroku with FastAPI** of ML DevOps Engineer Nanodegree Udacity
- Author: Mateus Cichelero
- Date: June 2022

## Link for the public github repository and heroku api docs

https://github.com/MateusCichelero/census_ml_deploy

https://census-salaries-prediction.herokuapp.com/docs#/

## Project Description

In this project, it is applied the skills to develop a classification model on publicly available Census Bureau data. Unit tests are created to monitor the model performance on various slices of the data. Then, the model is deployed using the FastAPI package and with API tests. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

This project builds the whole pipeline: data cleaning, model training, model validation and testing, tracking of data and artifacts with DVC (AWS S3 remote storage), and continuous integration (github actions, flake8, pytest) and deployment (heroku).

## Create environment
Use the supplied requirements file to create a new environment, or:

```bash
> conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
> conda activate [envname]
```

## Load artifacts
The projects artifacts are already created. With DVC installed and configured to remote storage, the artifacts can be downloaded by: 

```bash
> dvc pull -R
```

## Lint and Testing
After retrieving the artifacts, local testing can be performed by the following commands on the project root path: 

```bash
> flake8
> pytest
```

These commands are also performed in the CI github action in every update to the main branch.

## Using the API
Documentation for using the API can be found in: https://census-salaries-prediction.herokuapp.com/docs#/
It is also possible to locally call the external API by running the api_heroku_call.py script:

```bash
> python3 api_heroku_call.py
```

That command sends a POST request with the input data:

```js
{
    'age': 35,
    'workclass': 'Private',
    'fnlgt': 149184,
    'education': 'Masters',
    'marital_status': 'Divorced',
    'occupation': 'Prof-specialty',
    'relationship': 'Wife',
    'race': 'White',
            'sex': 'Female',
            'hoursPerWeek': 58,
            'nativeCountry': 'United-States'
}
```

And should return 200 as response code and '>50k' as prediction result.