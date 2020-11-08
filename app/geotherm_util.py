from app.py_output.Landsat_LST import Landsat_lst
import ee


ee.Initialize()

landsat = 'L8'
start = '2020-03-01'
end = '2020-05-31'
use_ndvi = True

# using US as a checking point
longitude = 37.3875
latittude = 122.0575


def get_geothermal_data(latittude=37.3875, longitude=122.0575):
    '''
    Returns single lst data based on given location.
    throws error if 
    '''
    point = ee.Geometry.Point([latittude, longitude])
    circle_region = point.buffer(10000)

    # Create ee request and call data

    lst = Landsat_lst.collection(landsat, start, end, circle_region, use_ndvi)

    value = lst.first().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Point([latittude, longitude]),
        maxPixels=1e13,
    ).get('LST').getInfo()

    return value


if __name__ == "__main__":
    get_geothermal_data(latittude, longitude)
