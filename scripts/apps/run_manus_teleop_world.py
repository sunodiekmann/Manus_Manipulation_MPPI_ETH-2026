import time
import numpy as np
import matplotlib.pyplot as plt

from scripts.streams.live_inputs import LiveInputs

from scripts.utils.config import (
    MANUS_PORT,
    LEFT_GLOVE_SERIAL,
    RIGHT_GLOVE_SERIAL,
    POSE_PORT,
    IMG_PORT,
    LEFT_BOARD_ID,
    RIGHT_BOARD_ID,
    CUBE_ID,
    LEFT_CALIB_JSON,
    RIGHT_CALIB_JSON,
    CUBE_SIZE,
    AXIS_LEN
)
from scripts.utils.calibration_io import load_calibration_matrix
from scripts.utils.transforms import pose_to_matrix, transform_points

from scripts.visualization.hand_plotting import (
    create_hand_artists,
    update_skeleton,
    clear_skeleton,
)
from scripts.visualization.frame_plotting import (
    create_frame_artists,
    update_frame_artists,
)
from scripts.visualization.object_plotting import (
    create_cube_artist,
    update_cube_mesh,
)

def main():
    live = LiveInputs(
        manus_port=MANUS_PORT,
        left_glove_serial_number=LEFT_GLOVE_SERIAL,
        right_glove_serial_number=RIGHT_GLOVE_SERIAL,
        pose_port=POSE_PORT,
        img_port=IMG_PORT,
        left_board_id=LEFT_BOARD_ID,
        right_board_id=RIGHT_BOARD_ID,
        cube_id=CUBE_ID,
        enable_images=False,
        debug_hz=1.0,
    )
    live.start()

    T_BR_left = load_calibration_matrix(LEFT_CALIB_JSON)
    T_BR_right = load_calibration_matrix(RIGHT_CALIB_JSON)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Gloves and Cube in World Frame")
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.1, 0.5)
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    left_scatter, left_lines, left_texts = create_hand_artists(
        ax, color="blue", label="Left Glove"
    )
    right_scatter, right_lines, right_texts = create_hand_artists(
        ax, color="red", label="Right Glove"
    )

    cube_poly = create_cube_artist(ax, alpha=0.25, facecolor="cyan", edgecolor="k")

    left_frame = create_frame_artists(ax, axis_len=AXIS_LEN)
    right_frame = create_frame_artists(ax, axis_len=AXIS_LEN)
    cube_frame = create_frame_artists(ax, axis_len=AXIS_LEN)

    ax.legend()

    try:
        while True:
            data = live.get_latest(max_age=0.2)

            # LEFT
            left_hand = data["left"]["skeleton"]
            left_board_pose = data["left_board_pose"]

            if left_hand is not None and left_board_pose is not None:
                T_WB_left = pose_to_matrix(left_board_pose)
                T_WR_left = T_WB_left @ T_BR_left
                left_pts_world = transform_points(T_WR_left, np.asarray(left_hand))
                update_skeleton(left_pts_world, left_scatter, left_lines, left_texts)
                update_frame_artists(left_frame, T_WR_left, "L wrist")
            else:
                clear_skeleton(left_scatter, left_lines, left_texts)

            # RIGHT
            right_hand = data["right"]["skeleton"]
            right_board_pose = data["right_board_pose"]

            if right_hand is not None and right_board_pose is not None:
                T_WB_right = pose_to_matrix(right_board_pose)
                T_WR_right = T_WB_right @ T_BR_right
                right_pts_world = transform_points(T_WR_right, np.asarray(right_hand))
                update_skeleton(right_pts_world, right_scatter, right_lines, right_texts)
                update_frame_artists(right_frame, T_WR_right, "R wrist")
            else:
                clear_skeleton(right_scatter, right_lines, right_texts)

            # CUBE
            cube_pose = data["cube_pose"]
            if cube_pose is not None:
                T_WC = pose_to_matrix(cube_pose)
                update_cube_mesh(cube_poly, T_WC, CUBE_SIZE)
                update_frame_artists(cube_frame, T_WC, "Cube")

            plt.pause(0.01)
            time.sleep(0.01)

    finally:
        live.stop()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()