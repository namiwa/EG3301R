'''
Contains the API logic for calling the interim model.
Only POST and GET request is valid from the API.
'''
import os
import io
import cv2
import werkzeug
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from flask_restful import Resource, abort, reqparse


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }
INTERIM_MODEL_DIR = 'interim_model'
IMG_PARAM = 'image'
IMAGE_SHAPE = (64, 64)
IMAGE_SIZE = 64
RGB_MAX = 255.0
CHANNEL = 3
CLASS_NAMES = ["Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
               "Industrial", "Pasture", "Permanent Crop", "Residential",
               "River", "Sea Lake"]


def allowed_file(filename):
    '''
    Returns if filename has valid image extension.
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class InterimApi(Resource):
    '''
    Main class for the interim API, bounded to interim/model URL path.
    '''

    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            IMG_PARAM, type=werkzeug.datastructures.FileStorage, location='files', required=True)
        super(InterimApi, self).__init__()

    def get(self):
        '''
        GET request to check tensorflow version of server.
        '''
        version = tf.version.VERSION
        return {'tf_version_interim': version}

    def post(self):
        '''
        Main function to send image file to server.
        Returns json with string result of the prediction.
        '''
        req = self.reqparse.parse_args()
        if IMG_PARAM not in req.keys():
            return abort(400)

        filename = secure_filename(req[IMG_PARAM].filename)
        valid = allowed_file(filename)
        if not valid:
            return abort(400)

        # Get interim image classification model path, and load model
        pwd = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(pwd, INTERIM_MODEL_DIR)
        # interim_model = tf.keras.models.load_model(model_dir)

        # Get image data from request, and store as 3D numpy array (64, 64, 3)
        image_bytes = req[IMG_PARAM].read()
        img_arr_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        img_arr_np = cv2.resize(img_arr_np, IMAGE_SHAPE)
        # Scale numpy array by max RGB Value
        img_arr_np = img_arr_np / RGB_MAX
        #img = np.array(img_arr_np).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL)

        # Run model prediction on request image
        #prediction = interim_model.predict(img)
        #prediction_index = np.argmax(prediction)
        #label = CLASS_NAMES[prediction_index]
        req[IMG_PARAM].close()

        return {'prediction': img_arr_np.tolist()[0] }
