import zmq

import time

import mujoco
import mujoco.viewer

import numpy as np

'''
This is a minimal Python example that talks to our MANUS C++ SDK and prints the data out directly to the terminal.
To run this, first run our MANUS SDK and then this script.

Keep in mind this data runs VERY fast, I would not recommend directly setting robot hands at this rate but instead slow down the data.
'''

left_glove_sn = "60f3738b"
right_glove_sn = "dc8b6ba8"

context = zmq.Context()
#Socket to talk to Manus SDK
print("Connecting to SDK")
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.CONFLATE, True)     
socket.connect("tcp://localhost:8000")

'''
This is ordered from Thumb to pinky, palm out to fingertip. 
The data is x,y,z for position and then quaternion (x,y,z,w) for rotation.  
25 positions * 7 for each pose = 175.  175 + 1 = 176, one datapoint for ID.

I highly recommend you visuzlize this data, it makes it much easier to figure this out.  :)
'''
def parse_full_skeleton(data):
    if data[0] == left_glove_sn:
        print("Left Glove Skeleton Data")
        print(list(map(float,data[1:])))
    elif data[0] == right_glove_sn:
        print("Right Glove Skeleton Data")
        print(list(map(float,data[1:])))
    else:
        print("Serial Number not found: " + str(data[0]))

model_path = "/home/valentin/git/postdoc/tesollo/output.xml"

model = mujoco.MjModel.from_xml_path(model_path)
mj_data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, mj_data)

def right_ergonomics_to_mujoco(ergonomics_data):
    ergonomics_data[1] = ergonomics_data[1] - 70.
    ergonomics_data[2] = ergonomics_data[2] + 45.

    for i in [1, 2, 3]:
        ergonomics_data[i * 4] = -ergonomics_data[i*4] + 0.

    tmp = ergonomics_data[0]
    ergonomics_data[0] = ergonomics_data[1]
    ergonomics_data[1] = tmp

    ergonomics_data[0] = -ergonomics_data[0]
    ergonomics_data[1] = -ergonomics_data[1]

    tmp = ergonomics_data[16]
    ergonomics_data[16] = ergonomics_data[17] * 0.5
    ergonomics_data[17] = -tmp

    ergonomics_data[16] = 0
    ergonomics_data[17] = 0
    ergonomics_data[18] = 0
    ergonomics_data[19] = 0

    return ergonomics_data

while viewer.is_running():
    #wait for message
    message = socket.recv()
    #receive the message from the socket
    message = message.decode('utf-8')
    # print("Received reply %s" % (message))
    data = message.split(",") 
    if len(data) == 40:
        print("Left Glove Joint-level Ergonomics Data:")
        print(list(map(float,data[0:20])))

        print("Right Glove Joint-level Ergonomics Data:")
        print(list(map(float,data[20:40])))

        right_hand_ergonomics = list(map(float,data[20:40]))
        right_hand_ergonomics = right_ergonomics_to_mujoco(right_hand_ergonomics)

        mj_data.qpos[:] = np.array(right_hand_ergonomics) / 360 * 2 * np.pi

        mujoco.mj_forward(model, mj_data)

    if viewer.is_running():
        viewer.sync()

    # elif len(data) == 352:
    #     parse_full_skeleton(data[0:176])
    #     parse_full_skeleton(data[176:352])
    # elif len(data) == 176:
    #     parse_full_skeleton(data[0:176])
    
    time.sleep(0.001)

viewer.close()
