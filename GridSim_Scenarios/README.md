# GridSim City Scenario

GridSim simulation engine inside an aerial map from Stockholm, Sweden.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The packages needed for install can be found inside requirements.txt: 

```
pip install -r requirements.txt
```

### Running the code

Menu function can be found in car_kinematic_city_menu.py
Run the script in order to access the menu where you can:

* play a simple run to see how GridSim works(simple button)
* record data(record button)
* replay a previously recorded run(replay button)

After you press the 'record' button you will have to complete the following paths:

* replay_path = the path where you want your replay data to be stored
* state_buf_path = the path where you want the car data to be stored(position, speed, steering angle and so on)
* all of the above paths must be completed, also you enter nonexistent paths(will be created at runtime)

In the near future we will implement features of recording images from the replay.  

# GridSim Highway Scenario  

GridSim simulation engine on a highway road, with highway traffic.

### Running the code  
  
Run the car_kinematic_highway script.
In order to add code to the script please override the custom function called in the main loop to avoid damaging the simulator flow.  

## Built with

* [Pygame](https://www.pygame.org/news) - A python programming language library for making multimedia applications like games built on top of the SDL library.
* [Tensorflow](https://www.tensorflow.org/) - An open source machine learning framework for everyone.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python.