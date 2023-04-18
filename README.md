# mmpush

Robot planar pushing code. This evolved out of the mm2d project.

## Install

```bash
# Clone the repository
git clone https://github.com/adamheins/mmpush

# Install basic dependencies
cd mmpush
poetry install

# Activate the virtualenv
poetry shell 
```

## Organization

* `mmpush`: Shared utilities.
* `scripts`: Scripts for pushing stuff.

## Experiments
Experiments are done using utilities in mobile_manipulation_central.

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
