import os
import ee
import requests
import pickle
import sklearn
import numpy as np
import pandas as pd
from flask_restful import Resource, abort, reqparse

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

        directory = os.path.dirname(__file__)
        path = os.path.join(directory, GEO_MODEL_DIR, GEO_MODEL_FILE)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        test = pd.DataFrame([[300, 'Dry steam']], columns=[
                            'Average LST', 'name_turbine_type'])
        predict = data.predict(test)

        return {'sklearn_version_geotherm': predict.item(0)}
