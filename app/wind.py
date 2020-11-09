'''
Contains the API logic for calling the wind model.
Only post request is valid from the API.
'''
import ee
import os
import json
import werkzeug
import requests
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask_restful import Resource, abort, reqparse


WIND_MODEL_DIR = 'wind_cnn_model'
LAT = 'latitude'
LONG = 'longtitude'
# Change this to suite the actual max power taking abs(max + min)
MAX_POWER = 1814.35805640538

ERA_DAILY = 'ECMWF/ERA5/DAILY'
FEATURE_COLS = ['mean_2m_air_temperature', 'surface_pressure',
                'u_component_of_wind_10m', 'v_component_of_wind_10m']
GENERATED_COLS = ['sat_wind_mag', 'sat_wind_ang']

# given 2 series like objects containing vectors, return its absolute direction
# arctan2 computed by numpy as radians, conversion to degrees


def wind_angle(u, v):
    return 180 + (180 / np.pi) * np.arctan2(v, u)

# given 2 series like objects containing vectors, return its absolute magnitude


def wind_magnitude(u, v):
    return np.sqrt(u*u + v*v)


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

    def _get_elevation_test(self, lat=1.0, lng=1.0):
        """
        Returns elevation height, based on geo location.
        """
        ee.Initialize()

        # Everset Dataset Test
        dem = ee.Image('USGS/SRTMGL1_003')
        xy = ee.Geometry.Point([lat, lng])
        elev = dem.sample(xy, 30).first().get('elevation').getInfo()
        print('Area elevation (m):', elev)

        return elev

    def _get_ERA_daily_frame(self, data=[]):
        df_image = pd.DataFrame([data], columns=FEATURE_COLS)

        # form calculated values
        df_image['sat_wind_mag'] = wind_magnitude(
            df_image['u_component_of_wind_10m'], df_image['v_component_of_wind_10m'])
        df_image['sat_wind_ang'] = wind_angle(
            df_image['u_component_of_wind_10m'], df_image['v_component_of_wind_10m'])

        min_max_scaler = MinMaxScaler()
        scaled_image = min_max_scaler.fit_transform(df_image.values)
        expanded_image = np.expand_dims(scaled_image, axis=-1)
        tensor_image = tf.convert_to_tensor(expanded_image, dtype_hint=float)

        return tensor_image

    def _get_ERA_daily(self, lat=1.0, lng=1.0):
        '''
        Returns raw satellite data from ERA daily averages.
        '''
        ee.Initialize()

        # Set ImageCollection Parameters for wind
        init_collect = ee.ImageCollection(ERA_DAILY)
        point = ee.Geometry.Point([lat, lng])
        time_range = init_collect.reduceColumns(
            ee.Reducer.minMax(), ["system:time_start"])

        # First request to get the last data set (most recent)
        end = ee.Date(time_range.get('max')).getInfo()['value']

        # EE call to get data
        collection = init_collect.filterDate(end).filterBounds(point)

        data = []
        for header in FEATURE_COLS:
            temp_data = collection.first().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                maxPixels=1e13,
            ).get(header).getInfo()
            data.append(temp_data)

        print(data)

        return data

    def get(self):
        """
        Returns tensorflow & Earth Engine version for wind api.
        """
        tf_version = tf.version.VERSION
        ee_version = ee.__version__
        return {'tf_version_wind': tf_version, 'ee_version_wind': ee_version}

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

        try:
            data = self._get_ERA_daily(lat, lng)
            data_arr = self._get_ERA_daily_frame(data)

            pwd = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(pwd, WIND_MODEL_DIR)
            wind_model = tf.saved_model.load(model_dir)

            prediction = wind_model(
                data_arr).numpy().flat[0].item() * MAX_POWER

            print(prediction)

            return {'prediction': prediction}

        except Exception as e:
            print(e)
            return {'error': "There was an error with your request"}
