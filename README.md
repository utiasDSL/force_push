# Force Push

Quasistatic robotic planar pushing with single-point contact using only force
feedback. 

## Install

This code has been tested on Ubuntu 20.04 with ROS Noetic and Python 3.8.
Initialize your catkin workspace before proceeding.

First install
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central)
into the catkin workspace.

Into the same catkin workspace, clone this repository:
```bash
cd ~/catkin_ws
git clone https://github.com/adamheins/force_push
```

Ensure all the Python dependencies in `requirements.txt` are satisfied (e.g.,
by doing something like `pip install -r requirements.txt`).

Build the workspace:
```bash
catkin build
```

## Simulation Experiments

Simulations are run in PyBullet. A small patch improving planar sliding
friction can be found
[here](https://github.com/bulletphysics/bullet3/pull/4539), which will require
you to build PyBullet from source:
```
# get patched version
git clone https://github.com/adamheins/bullet3

# build
cd bullet3
./build_cmake_pybullet_double.sh

# add this version to Python path
cd bullet3/build_cmake/examples/pybullet
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Run the simulations using the script
`scripts/simulation/pyb_simulation_many.py`. The results can be saved as a
Python pickle and post-processed using `scripts/simulation/process_sim_results.py`

## Hardware Experiments

Experiments are done using utilities in mobile_manipulation_central.

### Arm

If it isn't already, connect to the arm and put it into the required home
position. Then turn it off:
```
roslaunch mobile_manipulation_central thing.launch
rosrun mobile_manipulation_central home.py --config (rospack find force_push)/config/home.yaml --arm-only pushing_diag
```
Grasp a tennis ball with the gripper in the "pinched" configuration.

### Base, gripper, F/T sensor, Vicon

After the initial positioning of the arm, these experiments use only the mobile
base; the arm can remain off. SSH into the robot and run:
```
rosrun robotiq_ft_sensor rq_sensor
```
This will take a few seconds to detect the FT sensor, and will then start
publishing the `/robotiq_ft_wrench` topic.

On the laptop, run
```
rosrun mobile_manipulation_central ridgeback_vicon.launch
```
for localization and control of the mobile base.

### Calibration

For best results, the offset between the origin of the base frame (i.e., the
point about which the base rotates) and the contact point (i.e., roughly the
front of the tennis ball) should be calibrated. This can be done by temporarily
placing a marker on the tennis ball, ensuring Tracking is off in the Vicon UI,
and running the script `calibrate_contact_point.py`. This script automatically
looks for a marker near the expected location, calculates the offset, and
outputs the results to a YAML file. To use this calibration subsequently, move
the YAML file to the `config` directory.

It is also desirable to calibrate the orientation of the FT sensor (see the
calibration notes in the `mobile_manipulation_central` repository and the
scripts in `scripts/experiment/calibration`).

### Pushing controller

To run the pushing controller, use the script
`scripts/experiments/push_control_node.py` with desired options.

## License

MIT - see the LICENSE file.
