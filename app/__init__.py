#!flask/bin/python
'''
Main entry point for EG3301 Flask APIs using flask-restful.
'''
from flask import Flask
from flask_restful import Api

from app.interim import InterimApi
from app.wind import WindApi

APP = Flask(__name__)
API = Api(APP)


API.add_resource(InterimApi, '/interim', endpoint='interim')
API.add_resource(WindApi, '/predictions/wind', endpoint='wind')


if __name__ == "__main__":
    APP.run(debug=True)
