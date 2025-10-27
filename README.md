# CRL (Tesollo) robohand

This repo contains all necessary stuff for setting up the required things for the robotic hand we are using at CRL:

- Manus teleop
- IK stuff for UR5 teleop
- UR5 controller
- Pose estimation for simple boxes
- ROS2 stuff for actually controlling the robot
- Mujoco for simulation

# Getting started

Skip whatever you do not need below.

We will put all the things below in separate subfolders.
Start by cloning this repo, and running

```bash
cd <where_you_cloned_this>
# mkdir scripts
# mkdir core
git clone ..
mkdir manus

```

## Tesollo robot hand
To control the robot hand, we need to set up the sources for ROS2:
- [follow the instructions linked here](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

We will need the following packages from ROS2 that we can then install via apt if the setup above was successful:
```bash
sudo apt install ros-humble-desktop-full
sudo apt install ros-dev-tools
# sudo apt install ros-humble-xacro
# sudo apt install ros-humble-launch-ros
# sudo apt install ros-humble-hardware-interface
# sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

after installation, source the env
```bash
source /opt/ros/humble/setup.bash
```
We then set up the ROS2 stuff for our robot hand. You'll need to clone the repo https://github.com/Tesollo-Delto/DELTO_M_ROS2/tree/humble, and follow the instructions for setup.
Particularly:

```bash
mkdir tesollo
cd tesollo
git clone git@github.com:Tesollo-Delto/DELTO_M_ROS2.git
cd ..
rosdep install --from-paths tesollo --ignore-src -r -y
```

and after 

```bash
colcon build
source install/setup.bash
```

You'll likely have to add network rules to enable communication with the hand. 

To test, run the test-script supplied by Tesollo:

```bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py
```

and in a separate terminal

```bash
ros2 run dg5f_driver dg5f_right_test.py
```

## Manus

We have Manus Teleop gloves. From the teleop gloves, we get two types of data: [ergonomics](https://docs.manus-meta.com/2.4.0/Software/Rec/Ergonomics/), and [skeleton data](https://docs.manus-meta.com/2.5.0/Software/Skeletons/).

To install, we follow the [explanation here](https://docs.manus-meta.com/latest/Resources/), and download the `MANUS Core 3 SDK (including ROS2 Package)`.
You can then follow the [installation instructions here](https://docs.manus-meta.com/latest/Plugins/SDK/getting%20started/), and more specifically, the [Linux instructions here](https://docs.manus-meta.com/latest/Plugins/SDK/Linux/).

Generally, we will run the Manus Software in `integrated` mode.

After installing the required packages, you can build the LinuxClients using 
```bash
make all
```
in the relevant folder.

The hands should be calibrated, but if you want to recalibrate, you can follow the [instructions here](https://docs.manus-meta.com/2.4.0/Products/XR%20Metagloves/Calibration/)

#### Bridge to python via ZMQ:
One possiblity to get the Manus datasomewhere useful is via zmq.
For that, someone modified the manus client slightly:
- https://github.com/leap-hand/Bidex_Manus_Teleop?tab=readme-ov-file


We can test if this was succesful by starting the modified client:
```bash
```

and then running 
```bash
```

to plot our hand from the skeleton data.
The result should look roughly like this:
...

#### Using ROS2 for Manus
- [ ] TODO: look at ROS2 node.


## Pose estimation for simple objects
We use aruco marker, and/or april tags for board pose estimation

## UR5 Robot control
We use urrtde for ur5 robot control
- https://sdurobotics.gitlab.io/ur_rtde/


## Mocap and IK for complete robot control
- ...
