'''
Contains the API logic for calling the interim model.
Only POST and GET request is valid from the API.
'''
import os
import io
import cv2
import werkzeug
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask_restful import Resource, abort, reqparse


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }
INTERIM_MODEL_DIR = 'interim_model'


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
            'image', type=werkzeug.datastructures.FileStorage, location='files', required=True)
        super(InterimApi, self).__init__()

    def get(self):
        '''
        GET request to check tensorflow version of server.
        '''
        version = tf.version.VERSION
        return {'task': 'Tensorflow Version: ' + version}

    def post(self):
        '''
        Main function to send image file to server.
        Returns json with string result of the prediction.
        '''
        req = self.reqparse.parse_args()
        if ('image' not in req.keys()):
            return abort(400)

        filename = secure_filename(req['image'].filename)
        valid = allowed_file(filename)
        if not valid:
            return abort(400)

        pwd = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(pwd, 'wind_cnn_model')
        model = tf.keras.models.load_model(model_dir)

        print(dir(req['image']))

        '''
        file_im = io.BytesIO()
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_flag = 1
        img = cv2.imdecode(data, color_flag)'''

        return {'prediction': str(model.summary()), 'cv2': valid}
