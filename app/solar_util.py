import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Define the lat, long of the location and the year
lat, lon, year = 33.2164, -97.1292, 2019
# You must request an NSRDB api key from the link above
api_key = os.getenv('NREL_API_KEY')
# Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
attributes = 'ghi,dhi,dni'
# Choose year of data
year = '2019'
# Set leap year to true or false. True will return leap day data if present, false will not.
leap_year = 'false'
# Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
interval = '30'
# Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
# NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
# local time zone.
utc = 'false'
# Your full name, use '+' instead of spaces.
your_name = os.getenv('NREL_NAME')
# Your reason for using the NSRDB.
reason_for_use = 'project+work'
# Your affiliation
your_affiliation = os.getenv('NREL_AFFLIATION')
# Your email address
your_email = os.getenv('NREL_EMAIL')
# Please join our mailing list so we can keep you up-to-date on new developments.
mailing_list = 'false'


def solar_util(lat, lon):
    '''
    Returns Global Horizontal Irradiance, Direct Horizontal Irradiance, and Direct Normal Irradiance from NRSDB as numpy ndarray.
    '''
    # Declare url string
    url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
        year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)

    print(url)
    # Return all but first 2 lines of csv to get data:
    df = pd.read_csv(url, skiprows=2)

    # Set the time index in the pandas dataframe:
    df = df.set_index(pd.date_range(
        '1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=525600/int(interval)))

    # Extracting the featurs for model input
    features = ['GHI', 'DNI', 'DHI']
    solar_df = df[features].mean()
    print(solar_df.describe())

    return solar_df.values


if __name__ == "__main__":
    solar_input = solar_util(lat, lon)
    print(type(solar_input))
    print(solar_input)
