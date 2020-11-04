'''
Main entry point for EG3301 Flask APIs using flask-restful.
'''
from flask import Flask
from flask_restful import Api

from interim import InterimApi

APP = Flask(__name__)
API = Api(APP)


API.add_resource(InterimApi, '/interim/model', endpoint='user')


if __name__ == "__main__":
    APP.run(debug=True)
