import os
import tensorflow as tf
from flask import Flask
from flask_restful import Api, Resource
from flask_restful import reqparse

from interim import InterimApi

app = Flask(__name__)
api = Api(app)


api.add_resource(InterimApi, '/interim/model', endpoint='user')


if __name__ == "__main__":
    app.run(debug=True)
