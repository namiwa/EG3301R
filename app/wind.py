'''
Contains the API logic for calling the wind model.
Only post request is valid from the API.
'''
import ee
import os
import werkzeug
import requests
import tensorflow as tf
from flask_restful import Resource, abort, reqparse


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }
INTERIM_MODEL_DIR = 'wind_cnn_model'
LAT = 'latitude'
LONG = 'longtitude'
# Change this to suite the actual max power taking abs(max + min)
MAX_POWER = 1814.35805640538

ERA_DAILY = ''
FEATURE_COLS = ['mean_2m_air_temperature', 'surface_pressure',
                'u_component_of_wind_10m', 'v_component_of_wind_10m']
GENERATED_COLS = ['sat_wind_mag', 'sat_wind_ang']


class WindApi(Resource):
    '''
    Main class for calling wind data set & wind models for prediction.
    Looks for information only based on its latitude and longtitude location.
    '''

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            LAT, type=float, location='args', required=True)
        self.reqparse.add_argument(
            LONG, type=float, location='args', required=True)

    def _get_everest_test(self, lat=1.0, long=1.0):
        """
        Returns satellite data from earth engine, based on location.
        """
        ee.Initialize()

        # Everset Dataset Test
        dem = ee.Image('USGS/SRTMGL1_003')
        xy = ee.Geometry.Point([86.9250, 27.9881])
        elev = dem.sample(xy, 30).first().get('elevation').getInfo()
        print('Mount Everest elevation (m):', elev)

        return elev

    def get_get_ERA_daily(self, lat=1.0, long=1.0):

        pass

    def get(self):
        """
        Returns tensorflow version for wind api.
        """
        version = tf.version.VERSION
        return {'tf_version_wind': version}

    def post(self):
        """
        Returns array of scalled power prediction. 
        """
        req = self.reqparse.parse_args()
        if not LAT in req.keys():
            return abort(400)

        if not LONG in req.keys():
            return abort(400)

        lat = req[LAT]
        lng = req[LONG]

        data = self._get_everest_test(lat, lng)

        return {'prediction': data}
