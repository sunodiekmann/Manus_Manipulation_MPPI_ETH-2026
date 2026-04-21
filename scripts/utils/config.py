MANUS_PORT = 8000
LEFT_GLOVE_SERIAL = "5186fac7"
RIGHT_GLOVE_SERIAL = "dc8b6ba8"

POSE_PORT = 5557
IMG_PORT = 5558

LEFT_BOARD_ID = "0"
RIGHT_BOARD_ID = "1"
CUBE_ID = "2"

MANUS_ADDR = f"tcp://localhost:{MANUS_PORT}"
POSE_ADDR = f"tcp://localhost:{POSE_PORT}"
IMG_ADDR = f"tcp://localhost:{IMG_PORT}"

LEFT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/left_wrist2board_calibration.json"
RIGHT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json"

CUBE_SIZE = 0.07   # meters
AXIS_LEN = 0.04    # meters