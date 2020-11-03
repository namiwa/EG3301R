from flask import Flask
from flask_restful import Api, Resource
import os
import tensorflow as tf
from flask_restful import reqparse


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }
INTERIM_MODEL_DIR = 'interim_model'

app = Flask(__name__)
api = Api(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return "Hello World"


class UserApi(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('file', type=str, location='json')
        super(UserApi, self).__init__()

    def get(self, id):
        return {'task': 'testing'}

    def post(self, id):
        pwd = os.path.dirname(os.path.abspath(__file__))
        model = tf.saved_model.load(os.path.join(pwd, INTERIM_MODEL_DIR))
        args = self.reqparse.parse_args()
        return {'path': args['file']}

    def delete(self, id):
        pass


api.add_resource(UserApi, '/users/<int:id>', endpoint='user')

if __name__ == "__main__":
    app.run(debug=True)
