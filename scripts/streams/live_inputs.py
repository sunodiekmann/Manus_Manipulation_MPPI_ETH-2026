import time
from typing import Any, Dict, List, Optional

from scripts.streams.manus_skeleton_receiver import ManusSkeletonReceiver
from scripts.streams.manus_ergonomics_receiver import ManusErgonomicsReceiver
from scripts.streams.board_pose_listener import BoardPoseListener
from scripts.streams.image_listener import ImageListener


class LiveInputs:
    """
    Aggregates the latest samples from all live streams.

    This does NOT try to hard-synchronize streams online.
    It simply returns the newest available data together with timestamps and ages.
    That is the right behavior for live teleop/debugging.
    """

    def __init__(
        self,
        manus_port: int = 8000,
        left_glove_serial_number: str = "5186fac7",
        right_glove_serial_number: str = "dc8b6ba8",
        pose_port: int = 5557,
        img_port: int = 5558,
        left_board_id: str = "0",
        right_board_id: str = "1",
        cube_id: str = "2",
        camera_ids: Optional[List[str]] = None,
        enable_images: bool = False,
        debug_hz: float = 1.0,
    ):
        self.manus_port = manus_port
        self.left_glove_serial_number = left_glove_serial_number
        self.right_glove_serial_number = right_glove_serial_number

        self.pose_addr = f"tcp://localhost:{pose_port}"
        self.img_addr = f"tcp://localhost:{img_port}"

        self.right_board_id = str(right_board_id)
        self.left_board_id = str(left_board_id)
        self.cube_id = str(cube_id)
        self.camera_ids = [str(cid) for cid in (camera_ids or [])]

        self.enable_images = enable_images
        self.debug_hz = debug_hz

        self.skeleton = ManusSkeletonReceiver(
            port=self.manus_port,
            left_glove_serial_number=self.left_glove_serial_number,
            right_glove_serial_number=self.right_glove_serial_number,
        )

        self.ergonomics = ManusErgonomicsReceiver()
        self.pose_listener = BoardPoseListener(box_pose_socket_address=self.pose_addr)
        self.image_listener = (
            ImageListener(image_socket_address=self.img_addr)
            if self.enable_images
            else None
        )

        self._started = False
        self._last_debug_time = 0.0

    def start(self) -> bool:
        ok = True

        ok &= self.ergonomics.start()
        ok &= self.pose_listener.start()

        if self.image_listener is not None:
            ok &= self.image_listener.start()

        self._started = ok
        return ok

    def stop(self) -> None:
        try:
            self.ergonomics.stop()
        except Exception:
            pass

        try:
            self.pose_listener.stop()
        except Exception:
            pass

        if self.image_listener is not None:
            try:
                self.image_listener.stop()
            except Exception:
                pass

        try:
            self.skeleton.close()
        except Exception:
            pass

        self._started = False

    @staticmethod
    def _age(ts: Optional[float], now: float) -> Optional[float]:
        if ts is None:
            return None
        return now - ts

    @staticmethod
    def _is_fresh(age: Optional[float], max_age: float) -> bool:
        return age is not None and age <= max_age

    def get_latest(self, max_age: float = 0.2, include_images: Optional[bool] = None) -> Dict[str, Any]:
        """
        Returns the latest available snapshot from all streams.

        max_age:
            Used only for freshness flags. It does not discard data automatically.
        """
        now = time.time()
        use_images = self.enable_images if include_images is None else include_images

        # Skeleton - left
        left_skeleton_data = self.skeleton.get(glove="left")
        left_skeleton = left_skeleton_data["result"]
        left_skeleton_timestamp = left_skeleton_data["timestamp"]
        left_fingertip_positions = self.skeleton.get_fingertip_pos(glove="left")
        left_fingertip_quats = self.skeleton.get_fingertip_quat(glove="left")

        # Skeleton - right
        right_skeleton_data = self.skeleton.get(glove="right")
        right_skeleton = right_skeleton_data["result"]
        right_skeleton_timestamp = right_skeleton_data["timestamp"]
        right_fingertip_positions = self.skeleton.get_fingertip_pos(glove="right")
        right_fingertip_quats = self.skeleton.get_fingertip_quat(glove="right")

        # Ergonomics - left
        left_ergonomics_angles = self.ergonomics.get_left_angles_in_rad()
        left_ergonomics_timestamp = self.ergonomics.get_left_timestamp()

        # Ergonomics - right
        right_ergonomics_angles = self.ergonomics.get_right_angles_in_rad()
        right_ergonomics_timestamp = self.ergonomics.get_right_timestamp()

        # Poses
        left_board_pose = self.pose_listener.get_pose(self.left_board_id)
        left_board_timestamp = self.pose_listener.get_timestamp(self.left_board_id)

        right_board_pose = self.pose_listener.get_pose(self.right_board_id)
        right_board_timestamp = self.pose_listener.get_timestamp(self.right_board_id)

        cube_pose = self.pose_listener.get_pose(self.cube_id)
        cube_timestamp = self.pose_listener.get_timestamp(self.cube_id)

        # Images
        images: Dict[str, Any] = {}
        image_timestamps: Dict[str, Optional[float]] = {}

        if use_images and self.image_listener is not None:
            cam_ids = self.camera_ids or self.image_listener.get_tracked_camera_ids()
            for cam_id in cam_ids:
                images[cam_id] = self.image_listener.get_frame(cam_id)
                image_timestamps[cam_id] = self.image_listener.get_last_timestamp(cam_id)

        ages = {
            "left_skeleton": self._age(left_skeleton_timestamp, now),
            "right_skeleton": self._age(right_skeleton_timestamp, now),
            "left_ergonomics": self._age(left_ergonomics_timestamp, now),
            "right_ergonomics": self._age(right_ergonomics_timestamp, now),
            "left_board": self._age(left_board_timestamp, now),
            "right_board": self._age(right_board_timestamp, now),
            "cube": self._age(cube_timestamp, now),
        }

        freshness = {
            "left_skeleton": self._is_fresh(ages["left_skeleton"], max_age),
            "right_skeleton": self._is_fresh(ages["right_skeleton"], max_age),
            "left_ergonomics": self._is_fresh(ages["left_ergonomics"], max_age),
            "right_ergonomics": self._is_fresh(ages["right_ergonomics"], max_age),
            "left_board": self._is_fresh(ages["left_board"], max_age),
            "right_board": self._is_fresh(ages["right_board"], max_age),
            "cube": self._is_fresh(ages["cube"], max_age),
        }

        result = {
            "timestamp": now,
            "left": {
                "skeleton": left_skeleton,
                "skeleton_timestamp": left_skeleton_timestamp,
                "fingertip_positions": left_fingertip_positions,
                "fingertip_quats": left_fingertip_quats,
                "ergonomics_angles": left_ergonomics_angles,
                "ergonomics_timestamp": left_ergonomics_timestamp,
            },
            "right": {
                "skeleton": right_skeleton,
                "skeleton_timestamp": right_skeleton_timestamp,
                "fingertip_positions": right_fingertip_positions,
                "fingertip_quats": right_fingertip_quats,
                "ergonomics_angles": right_ergonomics_angles,
                "ergonomics_timestamp": right_ergonomics_timestamp,
            },
            "left_board_pose": left_board_pose,
            "left_board_timestamp": left_board_timestamp,
            "right_board_pose": right_board_pose,
            "right_board_timestamp": right_board_timestamp,
            "cube_pose": cube_pose,
            "cube_timestamp": cube_timestamp,
            "ages": ages,
            "freshness": freshness,
        }

        if use_images:
            result["images"] = images
            result["image_timestamps"] = image_timestamps

        self._maybe_debug_print(result)
        return result

    def _maybe_debug_print(self, data: Dict[str, Any]) -> None:
        if self.debug_hz <= 0:
            return

        now = time.time()
        if now - self._last_debug_time < 1.0 / self.debug_hz:
            return

        self._last_debug_time = now

        ages = data["ages"]

        def fmt_age(name: str) -> str:
            age = ages.get(name)
            if age is None:
                return f"{name}=missing"
            return f"{name}={age:.3f}s"

        tracked_boards = self.pose_listener.get_tracked_board_ids()
        tracked_cams = []
        if self.image_listener is not None:
            tracked_cams = self.image_listener.get_tracked_camera_ids()

        msg = (
            f"[LiveInputs] "
            f"{fmt_age('left_skeleton')} | "
            f"{fmt_age('left_ergonomics')} | "
            f"{fmt_age('left_board')} | "
            f"{fmt_age('right_skeleton')} | "
            f"{fmt_age('right_ergonomics')} | "
            f"{fmt_age('right_board')} | "
            f"{fmt_age('cube')} | "
            f"boards={tracked_boards}"
        )

        if self.image_listener is not None:
            msg += f" | cams={tracked_cams}"

        print(msg)