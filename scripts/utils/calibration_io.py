import json
import numpy as np


def load_calibration_matrix(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "T_board_wrist_optimized" in data:
        key = "T_board_wrist_optimized"
    else:
        key = "T_board_wrist"

    print(f"[calibration] {json_path}: using '{key}'")
    return np.array(data[key]["matrix_4x4"], dtype=float)