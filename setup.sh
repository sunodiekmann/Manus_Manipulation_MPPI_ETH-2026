#!/bin/bash

BASE_DIR="/home/suno/robohand2"

echo "Starting Manus client..."
gnome-terminal -- bash -ic "
cd \"$BASE_DIR/src/manus/SDKClient_Linux\" || exit 1
expect -c '
spawn ./SDKClient_Linux.out
sleep 2
send \"1\r\"
interact
'
"

echo "Starting pose estimation..."
gnome-terminal -- bash -ic "
cd \"/home/suno/robohand2/src/tag-pose-estimation\" || { echo 'cd failed'; exec bash; }
source venv/bin/activate || { echo 'venv activate failed'; exec bash; }
python3 scripts/run_pose_estimation.py --config=\"/home/suno/robohand2/src/tag-pose-estimation/config/pose_estimation_configs/manus_pose_estimation_config.json\"
echo
echo 'Process exited with code '$?
exec bash
"