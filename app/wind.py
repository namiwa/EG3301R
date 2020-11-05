'''
Contains the API logic for calling the wind model.
Only post request is valid from the API.
'''
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


class WindApi(Resource):
    '''
    Main class for calling wind data set & wind models for prediction.
    Looks for information only based on its latitude and longtitude location 
    '''

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            LAT, type=float, location='args', required=True)
        self.reqparse.add_argument(
            LONG, type=float, location='args', required=True)

    def get(self):
        """
        Returns tensorflow version for wind api
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

        return {'prediction': 'wind'}
