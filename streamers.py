import glfw, mujoco
import time, threading
import numpy as np
from PySide6.QtCore import Signal, QObject

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class CameraStreamer(QObject):
    frame_ready = Signal(np.ndarray)
    detection_ready = Signal(str)

    def __init__(self, simulation, update_rate= CONFIG["camera_streamer"]["fps"]):
        super().__init__()
        self._simulation = simulation
        self._update_rate = update_rate
        self._terminated = False
        self._pause_event = threading.Event()
        self._predictor = simulation.orchestrator.predictor

        self.width = CONFIG["camera_streamer"]["width"]
        self.height = CONFIG["camera_streamer"]["height"]

        # Init visualization objects
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(ENVIRONMENT.model, mujoco.mjtObj.mjOBJ_CAMERA, "bottom_cam")

        LOGGER.info(f"\tCameraStreamer: Initiated {self.__class__.__name__}")

    def terminate(self):
        self._terminated = True

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialized")
        
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.SAMPLES, 4)
        offscreen_window = glfw.create_window(self.width, self.height, "", None, None)
        glfw.make_context_current(offscreen_window)
        scene = mujoco.MjvScene(ENVIRONMENT.model, maxgeom=1000)
        context = mujoco.MjrContext(ENVIRONMENT.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        while not self._terminated:
            if self._simulation.is_loop_state('pause'):
                continue

            mujoco.mjv_updateScene(ENVIRONMENT.model, ENVIRONMENT.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

            rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, self.width, self.height), scene, context)
            mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, self.width, self.height), context)
            rgb_image = np.flip(rgb_buffer, axis=0)
            self.frame_ready.emit(rgb_image)

            if self._predictor is not None:
                self._predictor.model.detect(frame= rgb_image)

            time.sleep(1.0 / self._update_rate)
