# CRL (Tesollo) robohand simulation

This repo contains a guide to setup the end-to-end pipeline to run teleop and capture movement sequences of hand-object interactions with Manus Quantumgloves and aruco tag-based pose estimation, retarget them to the Tesollo robohand using an IK-solver and replay the sequences in a Mujoco simulation with a simple feedforward controller.

# Getting started

General:

- Clone the repository
- Install submodules
- Install packages

Setup tag pose estimation:

- Create configs for the cameras being used, current setup works only for a specific setup
- Define world frame with object board
- Do single extrinsic calibration of all cameras individually or do a dual camera extrinsic calibration
- Create object boards that are being tracked, in setup left, right glove and a 6cm cube are tracked
- Visualize object boards
- Change pose estimation config file, you can enable images if you want to stream the images as well. current setup runs on ~15hz using two realsense cameras
- Now you can run the pose estimation script, it shows live stream of both cameras and tracked poses of all object boards in the terminal

Setup manus qunatumgloves:

- Make sure license usb and receiver is connected then you can turn on the gloves. once the led shines constanly blue they are connected
- Start the integrated manus client as follows to enable data streaming and verify functionality
- For a new user the calibration should be done again...

Setup teleop and retargeting:
- If everything has been setup correctly individually you can close everything open a new code terminal and run the bash script to start everything automatically.
- In scripts/apps you can find all executable scripts
- You need to do first a coare calibration to get the transformation from wrist to board
- Next run run_manus_teleop_writs.py to verify streaming of manus gloves. Afterwards run the world version. It plots in realtime the gloves and object in worldframe
- Once verified everything works you should do the finer calibration to optimze the hand-object alignment
- Now you can also run the teleop tesollo script to visualize the retargeted hand-object interaction livetime

Start recording:
- Run recording script
- Process it
- Plot it to verify
- Reply in mujoco
- Run physical simulation

