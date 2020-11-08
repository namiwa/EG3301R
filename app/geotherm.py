import os
import ee
import requests
import pickle
import sklearn
import numpy as np
import pandas as pd
from flask_restful import Resource, abort, reqparse

from app.geotherm_util import get_geothermal_data

GEO_MODEL_DIR = 'geothermal_model'
GEO_MODEL_FILE = 'geothermal_model.sav'
LAT = 'latitude'
LONG = 'longtitude'
TURBINE = 'turbine'


class GeothermalApi(Resource):
    '''
    Main class for calling solar dataset & model prediction
    '''

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            LAT, type=float, location='args', required=True)
        self.reqparse.add_argument(
            LONG, type=float, location='args', required=True)
        self.reqparse.add_argument(
            TURBINE, type=str, location='args', required=True)

    def get(self):
        '''
        Returns sklearn version.
        '''
        version = sklearn.__version__
        return {'sklearn_version': version}

    def post(self):
        '''
        Returns predicted power based on lat, lng & turbine type.
        '''
        req = self.reqparse.parse_args()

        turbine = req[TURBINE]
        lat = req[LAT]
        lng = req[LONG]

        if (lng < -180 or lng > 180):
            return {'error': 'Longtitude out of valid range (-180 to 180 degrees)'}

        if (lat < -85.05112878 or lat > 85.05112878):
            return {'error': 'Latittude out of valid range (-85.05112878 to 85.05112878 degrees)'}

        directory = os.path.dirname(__file__)
        path = os.path.join(directory, GEO_MODEL_DIR, GEO_MODEL_FILE)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        try:
            lst = get_geothermal_data(lat, lng)
            print(lst, 'lst from location by user')
            test = pd.DataFrame([[lst, turbine]], columns=[
                'Average LST', 'name_turbine_type'])

            predict = data.predict(test)

            return {'prediction': predict.item(0)}

        except Exception as e:
            print(e)
            return {'error': 'Please try a different location'}
