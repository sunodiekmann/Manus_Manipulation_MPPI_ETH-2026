import zmq
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

left_glove_sn = "60f3738b"
right_glove_sn = "dc8b6ba8"

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.CONFLATE, True)
socket.connect("tcp://localhost:8000")


# --- setup 3D figure ---
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("MANUS Glove Skeleton")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
ax.view_init(elev=30, azim=30)

# pre-create left/right scatter + lines + text labels
left_scatter = ax.scatter([], [], [], color='blue', label='Left Glove')
right_scatter = ax.scatter([], [], [], color='red', label='Right Glove')
left_lines = [ax.plot([], [], [], 'b-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
right_lines = [ax.plot([], [], [], 'r-', linewidth=1, alpha=0.5)[0] for _ in range(5)]
left_texts = [ax.text(0, 0, 0, '', color='blue', fontsize=7) for _ in range(25)]
right_texts = [ax.text(0, 0, 0, '', color='red', fontsize=7) for _ in range(25)]
ax.legend()

# --- helper functions ---
def update_skeleton(floats, scatter, lines, texts):
    pts = np.array(floats).reshape(-1, 7)[:, :3]
    pts[:, 0] = -pts[:, 0]  # mirror X for handedness correction
    
    scatter._offsets3d = (pts[:,0], pts[:,1], pts[:,2])

    # update finger lines
    for i in range(5):
        finger = np.vstack((pts[0], pts[1 + i*5:1 + (i+1)*5]))  # connect to root (joint 0)
        lines[i].set_data(finger[:,0], finger[:,1])
        lines[i].set_3d_properties(finger[:,2])
    
    # update labels
    for i, txt in enumerate(texts):
        txt.set_position((pts[i,0], pts[i,1]))
        txt.set_3d_properties(pts[i,2])
        txt.set_text(str(i))

def parse_full_skeleton(data):
    sn = data[0]
    floats = list(map(float, data[1:]))
    if sn == left_glove_sn:
        update_skeleton(floats, left_scatter, left_lines, left_texts)
    elif sn == right_glove_sn:
        update_skeleton(floats, right_scatter, right_lines, right_texts)
    else:
        print("Serial Number not found:", sn)

# --- main loop ---
while True:
    message = socket.recv().decode('utf-8')
    data = message.split(",")

    if len(data) == 352:
        parse_full_skeleton(data[0:176])
        parse_full_skeleton(data[176:352])
    elif len(data) == 176:
        parse_full_skeleton(data[0:176])

    if len(data) == 40:
        rh = list(map(float,data[20:40]))

    plt.pause(0.01)
    time.sleep(0.05)
