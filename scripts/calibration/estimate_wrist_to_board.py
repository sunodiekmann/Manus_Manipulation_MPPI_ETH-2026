import json
import time
import numpy as np

from scripts.streams.board_pose_listener import BoardPoseListener
from scripts.streams.manus_skeleton_receiver import ManusSkeletonReceiver

from scripts.utils.transforms import (
    pose_to_matrix,
    build_transform,
    matrix_to_dict,
    rigid_transform_3d,
)
from scripts.utils.config import (
    POSE_ADDR,
    MANUS_PORT,
    LEFT_BOARD_ID,
    RIGHT_BOARD_ID,
    LEFT_GLOVE_SERIAL,
    RIGHT_GLOVE_SERIAL,
)

GLOVE = "left"
BOARD_ID = LEFT_BOARD_ID if GLOVE == "left" else RIGHT_BOARD_ID

OUTPUT_JSON = (
    f"src/tag-pose-estimation/config/calibration_gloves/"
    f"{GLOVE}_wrist2board_calibration.json"
)

# These are target fingertip positions in WORLD frame.
WORLD_POINTS_LEFT = {
    "thumb": np.array([0.08, 0.00, 0.00]),
    "index": np.array([0.08, 0.08, 0.00]),
    "middle": np.array([0.04, 0.08, 0.00]),
}

WORLD_POINTS_RIGHT = {
    "thumb": np.array([0.00, 0.00, 0.00]),
    "index": np.array([0.00, 0.08, 0.00]),
    "middle": np.array([0.04, 0.08, 0.00]),
}

TIP_IDXS = {
    "thumb": 4,
    "index": 9,
    "middle": 14,
}


def get_target_world_points(glove: str) -> np.ndarray:
    world_points_dict = WORLD_POINTS_LEFT if glove == "left" else WORLD_POINTS_RIGHT
    return np.stack(
        [
            world_points_dict["thumb"],
            world_points_dict["index"],
            world_points_dict["middle"],
        ],
        axis=0,
    )


def print_capture_status(skeleton_wrist_frame: np.ndarray, glove: str) -> None:
    thumb_wrist = skeleton_wrist_frame[TIP_IDXS["thumb"]]
    index_wrist = skeleton_wrist_frame[TIP_IDXS["index"]]
    middle_wrist = skeleton_wrist_frame[TIP_IDXS["middle"]]

    measured_thumb_index = np.linalg.norm(thumb_wrist - index_wrist)
    measured_index_middle = np.linalg.norm(index_wrist - middle_wrist)
    measured_thumb_middle = np.linalg.norm(thumb_wrist - middle_wrist)

    target_world_points = get_target_world_points(glove)
    target_thumb_index = np.linalg.norm(target_world_points[0] - target_world_points[1])
    target_index_middle = np.linalg.norm(target_world_points[1] - target_world_points[2])
    target_thumb_middle = np.linalg.norm(target_world_points[0] - target_world_points[2])

    print("\nCurrent fingertip positions in wrist frame:")
    print("  thumb :", thumb_wrist)
    print("  index :", index_wrist)
    print("  middle:", middle_wrist)

    print("\nMeasured fingertip distances in wrist frame:")
    print(f"  thumb-index : {measured_thumb_index:.4f} m")
    print(f"  index-middle: {measured_index_middle:.4f} m")
    print(f"  thumb-middle: {measured_thumb_middle:.4f} m")

    print("\nExpected distances from target WORLD points:")
    print(f"  thumb-index : {target_thumb_index:.4f} m")
    print(f"  index-middle: {target_index_middle:.4f} m")
    print(f"  thumb-middle: {target_thumb_middle:.4f} m")

    print("\nAbsolute error:")
    print(f"  thumb-index : {abs(measured_thumb_index - target_thumb_index):.4f} m")
    print(f"  index-middle: {abs(measured_index_middle - target_index_middle):.4f} m")
    print(f"  thumb-middle: {abs(measured_thumb_middle - target_thumb_middle):.4f} m")


def average_transform_matrices(transform_list: list[np.ndarray]) -> np.ndarray:
    if len(transform_list) == 1:
        return transform_list[0].copy()

    translations = np.stack([T[:3, 3] for T in transform_list], axis=0)
    mean_translation = translations.mean(axis=0)

    rotations = np.stack([T[:3, :3] for T in transform_list], axis=0)
    mean_rotation_raw = rotations.mean(axis=0)
    U, _, Vt = np.linalg.svd(mean_rotation_raw)
    mean_rotation = U @ Vt

    if np.linalg.det(mean_rotation) < 0:
        U[:, -1] *= -1
        mean_rotation = U @ Vt

    mean_transform = np.eye(4)
    mean_transform[:3, :3] = mean_rotation
    mean_transform[:3, 3] = mean_translation
    return mean_transform


def main():
    board_pose_listener = BoardPoseListener(box_pose_socket_address=POSE_ADDR)
    manus_skeleton_receiver = ManusSkeletonReceiver(
        port=MANUS_PORT,
        left_glove_serial_number=LEFT_GLOVE_SERIAL,
        right_glove_serial_number=RIGHT_GLOVE_SERIAL,
    )

    board_pose_listener.start()

    try:
        print("Waiting for board pose and glove skeleton...")
        print(f"Glove: {GLOVE}")
        print(f"Board ID: {BOARD_ID}")
        print("Place the fingertips at the defined WORLD points.")
        print("Commands:")
        print("  c + Enter  -> capture current frame")
        print("  Enter      -> finish and solve")
        print("  q + Enter  -> quit\n")

        while True:
            board_pose_world = board_pose_listener.get_pose(BOARD_ID)
            glove_data = manus_skeleton_receiver.get(glove=GLOVE)

            if board_pose_world is not None and glove_data["result"] is not None:
                break

            time.sleep(0.01)

        captured_fingertips_wrist_frame = []
        captured_board_to_world_transforms = []

        while True:
            board_pose_world = board_pose_listener.get_pose(BOARD_ID)
            glove_data = manus_skeleton_receiver.get(glove=GLOVE)

            if board_pose_world is None or glove_data["result"] is None:
                print("Data temporarily unavailable. Waiting...")
                time.sleep(0.1)
                continue

            skeleton_wrist_frame = glove_data["result"]
            print_capture_status(skeleton_wrist_frame, GLOVE)

            cmd = input(
                "\nCommand [c = capture, Enter = finish, q = quit]: "
            ).strip().lower()

            if cmd == "q":
                print("Aborted.")
                return

            if cmd == "":
                if len(captured_fingertips_wrist_frame) == 0:
                    print("No frames captured yet. Capture at least one frame first.")
                    continue
                break

            if cmd == "c":
                fingertips_wrist_frame = np.stack(
                    [
                        skeleton_wrist_frame[TIP_IDXS["thumb"]],
                        skeleton_wrist_frame[TIP_IDXS["index"]],
                        skeleton_wrist_frame[TIP_IDXS["middle"]],
                    ],
                    axis=0,
                )

                T_world_from_board = pose_to_matrix(board_pose_world)

                captured_fingertips_wrist_frame.append(fingertips_wrist_frame)
                captured_board_to_world_transforms.append(T_world_from_board)

                print(f"Captured frame {len(captured_fingertips_wrist_frame)}.")
                continue

            print("Unknown command.")

        mean_fingertips_wrist_frame = np.mean(
            np.stack(captured_fingertips_wrist_frame, axis=0),
            axis=0,
        )
        mean_T_world_from_board = average_transform_matrices(
            captured_board_to_world_transforms
        )
        target_fingertips_world_frame = get_target_world_points(GLOVE)

        rotation_world_from_wrist, translation_world_from_wrist = rigid_transform_3d(
            mean_fingertips_wrist_frame,
            target_fingertips_world_frame,
        )
        T_world_from_wrist = build_transform(
            rotation_world_from_wrist,
            translation_world_from_wrist,
        )

        # Convert wrist pose from world frame into board frame:
        # T_board_from_wrist = T_board_from_world @ T_world_from_wrist
        T_board_from_wrist = np.linalg.inv(mean_T_world_from_board) @ T_world_from_wrist

        calibration_result = {
            "description": (
                "Calibration transform from wrist frame to board frame "
                "using averaged multi-frame 3-point fingertip correspondences "
                "to fixed WORLD-frame target points"
            ),
            "board_id": BOARD_ID,
            "glove": GLOVE,
            "tip_indices": TIP_IDXS,
            "num_captures": len(captured_fingertips_wrist_frame),
            "target_world_points": {
                "thumb": target_fingertips_world_frame[0].tolist(),
                "index": target_fingertips_world_frame[1].tolist(),
                "middle": target_fingertips_world_frame[2].tolist(),
            },
            "mean_fingertips_wrist_frame": {
                "thumb": mean_fingertips_wrist_frame[0].tolist(),
                "index": mean_fingertips_wrist_frame[1].tolist(),
                "middle": mean_fingertips_wrist_frame[2].tolist(),
            },
            "T_world_from_board": matrix_to_dict(mean_T_world_from_board),
            "T_world_from_wrist": matrix_to_dict(T_world_from_wrist),
            "T_board_from_wrist": matrix_to_dict(T_board_from_wrist),
            # keep old key too for compatibility
            "T_board_wrist": matrix_to_dict(T_board_from_wrist),
        }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(calibration_result, f, indent=2)

        print("\nCalibration saved to:", OUTPUT_JSON)
        print("\nT_board_from_wrist =")
        print(np.array_str(T_board_from_wrist, precision=6, suppress_small=True))

    finally:
        board_pose_listener.stop()
        manus_skeleton_receiver.close()


if __name__ == "__main__":
    main()