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

We will put all the things below in separate subfolders, and we track the development of the separate things using git submodules.
Start by cloning this repo, and running

```bash
cd <where_you_cloned_this>
git submodule update --init --recursive
uv sync
```

## Tesollo robot hand
To control the robot hand, we need to set up the sources for ROS2:
- [follow the instructions linked here](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

We will need the following packages from ROS2 that we can then install via apt if the setup above was successful.
I think the commented things are not required, but when you get an error after installation, these are a good start.
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
cd ros_ws
rosdep install --from-paths src/ --ignore-src -r -y
```

and after 

```bash
colcon build
source install/setup.bash
```

You'll might have to add network rules to enable communication with the hand:
You'll have to add a network profile with address `169.254.186.0` and netmask `255.255.255.0` to enable communication with the robot.

To test, run the test-script supplied by Tesollo (in the ros_ws-dir):

```bash
ros2 launch dg5f_driver dg5f_right_driver.launch.py
```

and in a separate terminal

```bash
ros2 run dg5f_driver dg5f_right_test.py
```

## Manus

We have Manus Teleop gloves. From the teleop gloves, we get two types of data: [ergonomics](https://docs.manus-meta.com/2.4.0/Software/Rec/Ergonomics/), and [skeleton data](https://docs.manus-meta.com/2.5.0/Software/Skeletons/).

We work with the `MANUS Core 3 SDK (including ROS2 Package)`, which can be [found here](https://docs.manus-meta.com/latest/Resources/), or in the `src/manus` folder.

Since we apply some changes, we maintain aversion on github, which was cloned as a submodule above.

Ssince some of the files included (specifically, all the `.so` files) are too big to be tracked in github directly, we use git lfs to track them.

Therefore, in addition to the instructions below, we need to go into the manus-folder, and run `git lfs pulll`.

The original installation instructions are still applicable:
The original instructions can be found [on their website](https://docs.manus-meta.com/latest/Plugins/SDK/getting%20started/), and more specifically, the [Linux instructions](https://docs.manus-meta.com/latest/Plugins/SDK/Linux/).

Generally, we will run the Manus Software in `integrated` mode, but we still need to install grpc/protobuf.

After installing the required packages, you can build the LinuxClients using 
```bash
make all
```
in the linux-folders of the sdk.

The hands should be calibrated, but if you want to recalibrate, you can follow the [instructions here](https://docs.manus-meta.com/2.4.0/Products/XR%20Metagloves/Calibration/)

#### Bridge to python via ZMQ:
The change mentioned above is the introduction of communication via zmq, inspired by https://github.com/leap-hand/Bidex_Manus_Teleop?tab=readme-ov-file

We can test if the manus part works by starting the modified client, and pressing `1` for starting integrated mode.

In a separate terminal we can then run 

```bash
uv run scripts/plot.py
```
to plot our hand from the skeleton data.

#### Using ROS2 for Manus
- [ ] TODO: look at ROS2 node.

## Retargeting

#### GeoRT

We also cloned [GeoRT](https://github.com/facebookresearch/GeoRT).

## Pose estimation for simple objects
We use aruco marker, and/or april tags for board pose estimation

## UR5 Robot control
We use urrtde for ur5 robot control
- https://sdurobotics.gitlab.io/ur_rtde/


## Mocap and IK for complete robot control
- ...
