import os
import random
import requests
import pickle
import sklearn
import numpy as np
import pandas as pd
from flask_restful import Resource, abort, reqparse

SOLAR_MODEL_DIR = 'solar_model'
SOLAR_MODEL_FILE = 'solar_power_prediction.pkl'

FEATURE_HEADER = ['Global Horizontal Irradiance (GHI) | (W/m2)',
                  'Direct Normal Irradiance (DNI) | (W/m2)',
                  'Diffuse Horizontal Irradiance (DHI) | (W/m2)']

LAT = 'latitude'
LONG = 'longtitude'


class SolarApi(Resource):
    '''
    Main class for calling solar dataset & model prediction
    '''

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            LAT, type=float, location='args', required=True)
        self.reqparse.add_argument(
            LONG, type=float, location='args', required=True)

    def get(self):
        """
        Returns tensorflow version for wind api.
        """
        sk_version = sklearn.__version__

        return {'sklearn_version_solar': sk_version}

    def post(self):

        directory = os.path.dirname(__file__)
        path = os.path.join(directory, SOLAR_MODEL_DIR, SOLAR_MODEL_FILE)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Currently this is mocked before settling on a database backend
        test = pd.DataFrame(
            [[random.random() * 50 for i in range(len(FEATURE_HEADER))]], columns=FEATURE_HEADER)

        predict = data.predict(test)
        print(predict)

        return {'prediction': predict.item(0)}
