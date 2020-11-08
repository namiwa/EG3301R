import os
from geemap.conversion import js_to_python_dir


curr_dir = os.path.dirname(__file__)

in_dir = os.path.join(curr_dir, 'sofiaermida')
out_dir = os.path.join(curr_dir, 'py_output')

js_to_python_dir(in_dir, out_dir=out_dir, use_qgis=False)
