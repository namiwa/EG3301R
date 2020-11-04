import os
from flask import Flask
from flask_restful import Api, Resource
import tensorflow as tf
from flask_restful import reqparse

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }
INTERIM_MODEL_DIR = 'interim_model'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class InterimApi(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('file', type=str, location='json')
        super(InterimApi, self).__init__()

    def get(self):
        version = tf.version.VERSION
        return {'task': 'Tensorflow Version: ' + version}

    def post(self):
        pwd = os.path.dirname(os.path.abspath(__file__))
        model = tf.saved_model.load(os.path.join(pwd, INTERIM_MODEL_DIR))
        args = self.reqparse.parse_args()
        return {'path': args['file']}
