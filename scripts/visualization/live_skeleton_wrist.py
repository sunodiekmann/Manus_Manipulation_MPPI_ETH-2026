import time
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
)
from scripts.visualization.hand_plotting import (
    create_hand_artists,
    update_skeleton,
    clear_skeleton,
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

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("MANUS Glove Skeleton in Wrist Frame")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    left_scatter, left_lines, left_texts = create_hand_artists(
        ax,
        color="blue",
        label="Left Glove",
    )
    right_scatter, right_lines, right_texts = create_hand_artists(
        ax,
        color="red",
        label="Right Glove",
    )

    ax.legend()

    try:
        while True:
            data = live.get_latest(max_age=0.2)

            left_skeleton = data["left"]["skeleton"]
            right_skeleton = data["right"]["skeleton"]

            if left_skeleton is not None:
                update_skeleton(left_skeleton, left_scatter, left_lines, left_texts)
            else:
                clear_skeleton(left_scatter, left_lines, left_texts)

            if right_skeleton is not None:
                update_skeleton(right_skeleton, right_scatter, right_lines, right_texts)
            else:
                clear_skeleton(right_scatter, right_lines, right_texts)

            plt.pause(0.01)
            time.sleep(0.02)

    finally:
        live.stop()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()