from flask import Flask
from flask_restful import Api, Resource
import os
import tensorflow as tf

app = Flask(__name__)
api = Api(app)


@app.route('/')
def index():
    return "Hello World"


class UserApi(Resource):
    def get(self, id):
        return {'task': 'testing'}

    def put(self, id):
        pwd = os.getcwd()
        model = tf.saved_model.load(os.path.join(pwd, 'app'))
        return {'path': str(dir(model))}

    def delete(self, id):
        pass


api.add_resource(UserApi, '/users/<int:id>', endpoint='user')

if __name__ == "__main__":
    app.run(debug=True)
