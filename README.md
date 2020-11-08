# EG3301R

## ASI-304: Applying AI and Machine Learning to Satellite Data

This repo documents our technical exploration of applying AI and ML techniques on satellite data. See our prototype model running on Tensorflow.js [here.](https://asi-304-prototype.netlify.app/#/)

### Installation

- Clone this repo into a local directory
- Install Anaconda/Miniconda depending on your preference
- Create a Python virtual environment, minimally with Python version 3.6
- Activate the conda virtual environment, and install the following dependancies via pip install :
  ```
  pip install numpy pandas seaborn tensorflow-gpu opencv-python jupyter notebook
  ```
- Download the [Eurosat Dataset](https://github.com/phelber/EuroSAT) and unzip the EuroSat folder into the root directory of the local repo
- Attempt to run the jupyter notebooks to observe if the install had been successful

### Flask Backend Server

#### Core Server Dependacies

- Flask
- Flask-Restful
- Tensorflow
- Numpy
- Python-OpenCV
- Sci-Kit Learn version 0.22.2-1post to match Gooogle Collab version
- Dotenv
- Google Earth Engine Python Backend
  - [View installation steps for python backend](https://developers.google.com/earth-engine/guides/python_install)

#### Managing Conda environment

The root environment files contain the needed packages to run some of the jupyter notebooks found here. **Note that the version of python needed is 3.7.**

### Tensorflow Environment

No other configuration needed

### Solaris Environment

Clone the [Solaris](https://github.com/CosmiQ/solaris) repository locally. Set up the solaris conda environment using the yml file above, and change directory to the cloned repository. Run `pip install .` within the solaris repository to add solaris to the virtual environment.

## References

As we have used the [Geemap](https://github.com/giswqs/geemap) python package, we reference the work of Qiusheng Wu:

[1] **Wu, Q.**, (2020). geemap: A Python package for interactive mapping with Google Earth Engine. _The Journal of Open Source Software_, 5(51), 2305. https://doi.org/10.21105/joss.02305

As we have used the [Eurosat Dataset](https://github.com/phelber/EuroSAT):

[2] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

```
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```

[3] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

```
@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
```

As we have used the following earth engine module to calculate Land Surface Temperature:

[4] Ermida, S.L., Soares, P., Mantas, V., Göttsche, F.-M., Trigo, I.F., 2020.
Google Earth Engine open-source code for Land Surface Temperature estimation from the Landsat series.
'Remote Sensing, 12 (9), 1471; https':#doi.Org/10.3390/rs12091471
