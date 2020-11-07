import os
import requests
import pickle
import sklearn
import numpy as np
import pandas as pd
from flask_restful import Resource, abort, reqparse

SOLAR_MODEL_DIR = 'solar_model'
SOLAR_MODEL_FILE = 'solar_power_prediction_model.pkl'


class SolarApi(Resource):
    '''
    Main class for calling solar dataset & model prediction
    '''

    def get(self):
        """
        Returns tensorflow version for wind api.
        """
        version = sklearn.__version__

        directory = os.path.dirname(__file__)
        path = os.path.join(directory, SOLAR_MODEL_DIR, SOLAR_MODEL_FILE)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        test = pd.DataFrame([[50 for i in range(9)]], columns=['Weather file beam irradiance | (W/m2)',
                                                               'Weather file diffuse irradiance | (W/m2)',
                                                               'Weather file global horizontal irradiance | (W/m2)',
                                                               'Weather file ambient temperature | (C)',
                                                               'Transmitted plane of array irradiance | (W/m2)',
                                                               'Module temperature | (C)',
                                                               'Sun up over horizon | (0/1)',
                                                               'Plane of array irradiance | (W/m2)',
                                                               'Angle of incidence | (deg)'])

        predict = data.predict(test)
        print(predict)

        return {'sklearn_version_solar': str(data)}

    def post(self):
        return abort(400)
