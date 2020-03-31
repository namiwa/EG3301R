# EG3301R

## ASI-304: Applying AI and Machine Learning to Satellite Data

### Introduction
This repo documents our technical exploration of applying AI and ML techniques on satellite data.

### Installation

* Clone this repo into a local directory
* Install Anaconda/Miniconda depending on your preference
* Create a Python virtual environment, minimally with Python version 3.6
* Activate the conda virtual environment, and install the following dependancies via pip install : 
  ``` 
  pip install numpy pandas seaborn tensorflow-gpu opencv-python jupyter notebook
  ```
* Download the [Eurosat Dataset](https://github.com/phelber/EuroSAT) and unzip the EuroSat folder into the root directory of the local repo
* Attempt to run the jupyter notebooks to observe if the install had been successful

#### Managing Conda environment
The root environment files contain the needed packages to run some of the jupyter notebooks found here. **Note that the version of python needed is 3.7.**

##### Tensorflow Environment
No other configuration needed

##### Solaris Environment
Clone the [Solaris](https://github.com/CosmiQ/solaris) repository locally. Set up the solaris conda environment using the yml file above, and change directory to the cloned repository. Run ```pip install .``` within the solaris repository to add solaris to the virtual environment.

### References

As we have used the [Eurosat Dataset](https://github.com/phelber/EuroSAT): 

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

```
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

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
