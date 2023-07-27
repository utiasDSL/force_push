# Force Push

Quasistatic robotic planar pushing with single-point contact using only force
feedback. 

## Install

First install
[mobile_manipulation_central](https://github.com/utiasDSL/mobile_manipulation_central)
into the catkin workspace.

Into the same catkin workspace, clone this repository:
```bash
cd ~/catkin_ws
git clone https://github.com/adamheins/force_push
```

Build the workspace:
```bash
catkin build
```

## Experiments
Experiments are done using utilities in mobile_manipulation_central.

If it isn't already, connect to the arm and put it into the required home
position. Then turn it off:
```
roslaunch mobile_manipulation_central thing.launch
rosrun mobile_manipulation_central home.py --config (rospack find force_push)/config/home.yaml --arm-only pushing_diag
```
Grasp a tennis ball with the gripper in the "pinched" configuration.

These experiments use only the mobile base; no need to turn on the arm. SSH
into the robot and run:
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

## License

MIT - see the LICENSE file.
