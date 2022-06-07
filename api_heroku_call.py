"""
Heroku Api test script
"""
import requests

data = {
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


r = requests.post(
    'https://census-salaries-prediction.herokuapp.com/',
    json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
