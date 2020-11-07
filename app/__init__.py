#!flask/bin/python
'''
Main entry point for EG3301 Flask APIs using flask-restful.
'''
from flask import Flask
from flask_restful import Api

from app.interim import InterimApi
from app.wind import WindApi
from app.geotherm import GeothermalApi
from app.solar import SolarApi

APP = Flask(__name__)
API = Api(APP)


API.add_resource(InterimApi, '/interim', endpoint='interim')

API.add_resource(WindApi, '/predictions/wind', endpoint='wind')
API.add_resource(SolarApi, '/predictions/solar', endpoint='solar')
API.add_resource(GeothermalApi, '/predictions/geothermal',
                 endpoint='geothermal')


if __name__ == "__main__":
    APP.run(debug=True)
