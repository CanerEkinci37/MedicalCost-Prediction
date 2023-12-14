import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import json
import joblib

__model = None
__data_columns = None
__regions = None


def get_region_names():
    return __regions


def get_data_columns():
    return __data_columns


def load_artifacts():
    print("Loading saved artifacts...start")
    global __model
    with open("artifacts/saved_model.pkl", "rb") as f:
        __model = joblib.load(f)

    global __data_columns
    global __regions
    with open("artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __regions = __data_columns[5:]


def predict_medical_cost(age, bmi, children, ismale, issmoker, region):
    x = get_data_columns()
    loc_index = x.index(f"region_{region}")
    inputs = np.zeros(len(x))
    inputs[0], inputs[1], inputs[2], inputs[3], inputs[4] = (
        age,
        bmi,
        children,
        ismale,
        issmoker,
    )
    print(inputs)
    if loc_index >= 0:
        inputs[loc_index] = 1
    print(inputs)
    x_poly = PolynomialFeatures(degree=2).fit_transform([inputs])
    return round(__model.predict(x_poly)[0], 2)
