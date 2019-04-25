# kolt_ros

[![Build Status](https://travis-ci.org/diggerdata/yolov2_ros.svg?branch=master)](https://travis-ci.org/diggerdata/yolov2_ros) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) ![ros Melodic](https://img.shields.io/badge/ROS-Melodic-red.svg)

This package is a ROS implementation of the YOLOv2 algorithm using Keras and TensorFlow with a few added features, such as an object tracker.

**Keywords:** YOLOv2, Keras, TensorFlow

### License

The source code is released under a [BSD 3-Clause license](ros_package_template/LICENSE).

**Author(s):** Nathan Rosenberg  
**Maintainer:** Nathan Rosenberg, narosenberg@wpi.edu  
**Affiliation:** [Robotics Engineering, Worcester Polytechnic Institute](https://www.wpi.edu/academics/departments/robotics-engineering)  

The kolt_ros package has been tested under [ROS] Melodic and Ubuntu 18.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

### Documentation

You can find the documentation for KOLT on the [wiki](https://github.com/diggerdata/kolt_ros/wiki).


<!-- ![Example image](doc/example.jpg)


## Installation

### Installation from Packages

To install all packages from the this repository as Debian packages use

    sudo apt-get install ros-indigo-...

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org) (middleware for robotics),
- [Eigen] (linear algebra library)

		sudo apt-get install libeigen3-dev


#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd catkin_workspace/src
	git clone https://github.com/ethz-asl/ros_package_template.git
	cd ../
	catkin_make


### Unit Tests

Run the unit tests with

	catkin_make run_tests_ros_package_template


## Usage

Describe the quickest way to run this software, for example:

Run the main node with

	roslaunch ros_package_template ros_package_template.launch

## Config files

Config file folder/set 1

* **config_file_1.yaml** Shortly explain the content of this config file

Config file folder/set 2

* **...**

## Launch files

* **launch_file_1.launch:** shortly explain what is launched (e.g standard simulation, simulation with gdb,...) 
    
     Argument set 1

     - **`argument_1`** Short description (e.g. as commented in launch file). Default: `default_value`.

    Argument set 2

    - **`...`** 

* **...** 

## Nodes

### ros_package_template

Reads temperature measurements and computed the average.


#### Subscribed Topics

* **`/temperature`** ([sensor_msgs/Temperature])

	The temperature measurements from which the average is computed.


#### Published Topics

...


#### Services

* **`get_average`** ([std_srvs/Trigger])

	Returns information about the current average. For example, you can trigger the computation from the console with

		rosservice call /ros_package_template/get_average


#### Parameters

* **`subscriber_topic`** (string, default: "/temperature")

	The name of the input topic.

* **`cache_size`** (int, default: 200, min: 0, max: 1000)

	The size of the cache.


### NODE_B_NAME

...


## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/ethz-asl/ros_best_practices/issues).


[ROS]: http://www.ros.org
[rviz]: http://wiki.ros.org/rviz
[Eigen]: http://eigen.tuxfamily.org
[std_srvs/Trigger]: http://docs.ros.org/api/std_srvs/html/srv/Trigger.html
[sensor_msgs/Temperature]: http://docs.ros.org/api/sensor_msgs/html/msg/Temperature.html -->

