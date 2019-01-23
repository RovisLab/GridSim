# GridSim Configurable Map

GridSim simulation engine inside any desired map.
For this build we used an aerial map from Stockholm, Sweden.
Additional features:
	- mini-map
	- map and mini-map scaling factor
	- route tracking on mini-map

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The packages needed for install can be found inside requirements.txt: 

```
pip install -r requirements.txt
```

### Running the code

The main function can be found inside car_kinematic_model_sandbox.py.
In order to load a map, you have to change the MAP_PATH and OBJECT_MAP_PATH.
OBJECT_MAP has to be created on top of the original map by coloring any obstacle with a desired color(default is yellow(254, 242, 0))

## Built with

* [Pygame](https://www.pygame.org/news) - A python programming language library for making multimedia applications like games built on top of the SDL library.
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.