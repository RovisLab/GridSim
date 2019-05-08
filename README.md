# GridSim: A Vehicle Kinematics Engine for Deep Neuroevolutionary Control in Autonomous Driving

GridSim is an autonomous driving simulator engine that uses a car-like robot architecture to generate occupancy grids from simulated sensors.

[GridSim arXiv paper link](https://arxiv.org/abs/1901.05195)

Demo below: 

![Demo](https://github.com/RovisLab/GridSim/raw/master/GridSim_Scenarios/resources/gif/grid_sim_demo_as_gif.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Clone the repository:
```bash
$ git clone https://github.com/RovisLab/GridSim.git
```

### Prerequisites

The packages needed for install can be found inside requirements.txt: 

```
pip install -r requirements.txt
```

### Running the code

Each scenario cand be found in a separate folder:
* GridSim_City_Scenario: GridSim simulation engine inside an aerial map from Stockholm, Sweden.
* GridSim_Configurable_Map: GridSim simulation engine inside any desired map. For this build we used an aerial map from Stockholm, Sweden. Additional features: - mini-map - map and mini-map scaling factor - route tracking on mini-map
* GridSim_Seamless: GridSim simulation engine inside a seamless (never-ending) network of roads.

The main function can be found inside each folder, inside car_kinematic_model.py

## Built with

* [Pygame](https://www.pygame.org/news) - A python programming language library for making multimedia applications like games built on top of the SDL library.
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.

