import json
import time
from pathlib import Path

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

DURATION_SEC = 10.0

OUTPUT_ROOT = Path("recordings")
SESSION_NAME = time.strftime("session_%Y%m%d_%H%M%S")
SESSION_DIR = OUTPUT_ROOT / SESSION_NAME

HANDS_PATH = SESSION_DIR / "hands.jsonl"
ERGONOMICS_PATH = SESSION_DIR / "ergonomics.jsonl"
POSES_PATH = SESSION_DIR / "poses.jsonl"
META_PATH = SESSION_DIR / "meta.json"


def make_jsonable(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_jsonl_line(fp, obj):
    fp.write(json.dumps(obj) + "\n")
    fp.flush()


def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "session_name": SESSION_NAME,
        "start_wall_time": time.time(),
        "duration_sec": DURATION_SEC,
        "manus_port": MANUS_PORT,
        "pose_port": POSE_PORT,
        "img_port": IMG_PORT,
        "left_glove_serial_number": LEFT_GLOVE_SERIAL,
        "right_glove_serial_number": RIGHT_GLOVE_SERIAL,
        "left_board_id": LEFT_BOARD_ID,
        "right_board_id": RIGHT_BOARD_ID,
        "cube_id": CUBE_ID,
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

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

    last_left_hand_ts = None
    last_right_hand_ts = None

    last_left_ergo_ts = None
    last_right_ergo_ts = None

    last_left_board_ts = None
    last_right_board_ts = None
    last_cube_ts = None

    hand_count = 0
    ergonomics_count = 0
    pose_count = 0

    start_time = time.time()

    try:
        with (
            open(HANDS_PATH, "w") as hands_fp,
            open(ERGONOMICS_PATH, "w") as ergonomics_fp,
            open(POSES_PATH, "w") as poses_fp,
        ):
            print(f"Recording raw demo sequence into: {SESSION_DIR}")

            while True:
                now = time.time()
                if now - start_time >= DURATION_SEC:
                    break

                data = live.get_latest(max_age=1.0)
                did_write = False

                # -------------------------
                # LEFT HAND SKELETON
                # -------------------------
                left_hand_ts = data["left"]["skeleton_timestamp"]
                if left_hand_ts is not None and left_hand_ts != last_left_hand_ts:
                    sample = {
                        "stream": "left_hand",
                        "timestamp": left_hand_ts,
                        "record_time": now,
                        "joints_wrist": make_jsonable(data["left"]["skeleton"]),
                        "fingertip_positions_wrist": make_jsonable(data["left"]["fingertip_positions"]),
                        "fingertip_quats_wrist": make_jsonable(data["left"]["fingertip_quats"]),
                    }
                    write_jsonl_line(hands_fp, sample)
                    last_left_hand_ts = left_hand_ts
                    hand_count += 1
                    did_write = True

                # -------------------------
                # RIGHT HAND SKELETON
                # -------------------------
                right_hand_ts = data["right"]["skeleton_timestamp"]
                if right_hand_ts is not None and right_hand_ts != last_right_hand_ts:
                    sample = {
                        "stream": "right_hand",
                        "timestamp": right_hand_ts,
                        "record_time": now,
                        "joints_wrist": make_jsonable(data["right"]["skeleton"]),
                        "fingertip_positions_wrist": make_jsonable(data["right"]["fingertip_positions"]),
                        "fingertip_quats_wrist": make_jsonable(data["right"]["fingertip_quats"]),
                    }
                    write_jsonl_line(hands_fp, sample)
                    last_right_hand_ts = right_hand_ts
                    hand_count += 1
                    did_write = True

                # -------------------------
                # LEFT ERGONOMICS
                # -------------------------
                left_ergo_ts = data["left"]["ergonomics_timestamp"]
                if left_ergo_ts is not None and left_ergo_ts != last_left_ergo_ts:
                    sample = {
                        "stream": "left_ergonomics",
                        "timestamp": left_ergo_ts,
                        "record_time": now,
                        "angles_rad": make_jsonable(data["left"]["ergonomics_angles"]),
                    }
                    write_jsonl_line(ergonomics_fp, sample)
                    last_left_ergo_ts = left_ergo_ts
                    ergonomics_count += 1
                    did_write = True

                # -------------------------
                # RIGHT ERGONOMICS
                # -------------------------
                right_ergo_ts = data["right"]["ergonomics_timestamp"]
                if right_ergo_ts is not None and right_ergo_ts != last_right_ergo_ts:
                    sample = {
                        "stream": "right_ergonomics",
                        "timestamp": right_ergo_ts,
                        "record_time": now,
                        "angles_rad": make_jsonable(data["right"]["ergonomics_angles"]),
                    }
                    write_jsonl_line(ergonomics_fp, sample)
                    last_right_ergo_ts = right_ergo_ts
                    ergonomics_count += 1
                    did_write = True

                # -------------------------
                # LEFT BOARD
                # -------------------------
                left_board_ts = data["left_board_timestamp"]
                if left_board_ts is not None and left_board_ts != last_left_board_ts:
                    sample = {
                        "stream": "left_board",
                        "timestamp": left_board_ts,
                        "record_time": now,
                        "pose_world": make_jsonable(data["left_board_pose"]),
                    }
                    write_jsonl_line(poses_fp, sample)
                    last_left_board_ts = left_board_ts
                    pose_count += 1
                    did_write = True

                # -------------------------
                # RIGHT BOARD
                # -------------------------
                right_board_ts = data["right_board_timestamp"]
                if right_board_ts is not None and right_board_ts != last_right_board_ts:
                    sample = {
                        "stream": "right_board",
                        "timestamp": right_board_ts,
                        "record_time": now,
                        "pose_world": make_jsonable(data["right_board_pose"]),
                    }
                    write_jsonl_line(poses_fp, sample)
                    last_right_board_ts = right_board_ts
                    pose_count += 1
                    did_write = True

                # -------------------------
                # CUBE
                # -------------------------
                cube_ts = data["cube_timestamp"]
                if cube_ts is not None and cube_ts != last_cube_ts:
                    sample = {
                        "stream": "cube",
                        "timestamp": cube_ts,
                        "record_time": now,
                        "pose_world": make_jsonable(data["cube_pose"]),
                    }
                    write_jsonl_line(poses_fp, sample)
                    last_cube_ts = cube_ts
                    pose_count += 1
                    did_write = True

                if not did_write:
                    time.sleep(0.001)

        print(f"Done.")
        print(f"Saved {hand_count} hand events")
        print(f"Saved {ergonomics_count} ergonomics events")
        print(f"Saved {pose_count} pose events")
        print(f"hands      : {HANDS_PATH}")
        print(f"ergonomics : {ERGONOMICS_PATH}")
        print(f"poses      : {POSES_PATH}")
        print(f"meta       : {META_PATH}")

    except KeyboardInterrupt:
        print("Recording interrupted by user.")

    finally:
        live.stop()


if __name__ == "__main__":
    main()