from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import time

import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "output_mocap.xml"

FINGERTIP_INDICES = [0, 1, 2, 3, 4]
FINGERTIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

def plot_fingertips(data, mode='scatter'):
    # Compute shared axis limits
    xyz = data[:, FINGERTIP_INDICES, :].reshape(-1, 3)
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)

    fig = plt.figure(figsize=(15, 4))
    axs = [fig.add_subplot(1, 5, i+1, projection='3d') for i in range(5)]

    for ax, idx, name in zip(axs, FINGERTIP_INDICES, FINGERTIP_NAMES):
        pts = data[:, idx, :]
        if mode == 'scatter':
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, alpha=0.6)
        elif mode == 'line':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.5)
        else:
            raise ValueError("mode must be 'scatter' or 'line'")

        ax.set_title(name)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]

    N = 2000

    id_to_name = {}

    for i in range(model.ngeom):
        id_to_name[i] = model.geom(i).name

    limits = np.zeros((2, 20))

    for i in range(20):
        limits[0, i] = model.jnt_range[i, 0]  # lower limit
        limits[1, i] = model.jnt_range[i, 1]  # upper limit

    all_poses = []

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # configuration.update_from_keyframe("grasp hard")

        # # Initialize mocap bodies at their respective sites.
        # posture_task.set_target_from_configuration(configuration)
        # for finger in fingers:
        #     mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "geom")

        # rate = RateLimiter(frequency=500.0, warn=False)
        # dt = rate.dt
        # t = 0
        for _ in range(N):
          rnd = np.random.uniform(low=limits[0, :], high=limits[1, :])
          data.qpos[:] = rnd
          data.qvel[:] = 0
          
          mujoco.mj_forward(model, data)

          # If any contact distance < 0, collision
          coll_with_palm = False
          for i in range(data.ncon):
              if data.contact[i].dist < 0:
                  if "palm" in id_to_name[data.contact[i].geom1] or "palm" in id_to_name[data.contact[i].geom2]:
                      coll_with_palm = True
                      break

          if coll_with_palm:
              continue

          finger_pos = []
          for finger in fingers:
              pos=data.geom(finger).xpos.copy()
              finger_pos.append(pos)

          all_poses.append(finger_pos)

          viewer.sync()

          # time.sleep(0.1)

        viewer.close()

    all_poses = np.array(all_poses)

    plot_fingertips(all_poses)
