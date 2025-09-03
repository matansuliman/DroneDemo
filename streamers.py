import glfw
import time
import mujoco
import threading
import numpy as np
from PySide6.QtCore import Signal, QObject

import logging
logger = logging.getLogger("app")

FPS = 60
WIDTH  = 320
HEIGHT = 240


class CameraStreamer(QObject):
    frame_ready = Signal(np.ndarray)
    detection_ready = Signal(str)

    def __init__(self, simulation, update_rate =FPS):
        super().__init__()
        self._env = simulation.env
        self._simulation = simulation
        self._update_rate = update_rate
        self._terminated = False
        self._pause_event = threading.Event()
        self._predictor = simulation.orchestrator.predictor

        self.width = WIDTH
        self.height = HEIGHT

        # Init visualization objects
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_CAMERA, "bottom_cam")

        logger.info(f"\tCameraStreamer: Initiated {self.__class__.__name__}")

    def terminate(self):
        self._terminated = True

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialized")
        
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # <- Hide the window
        glfw.window_hint(glfw.SAMPLES, 4)
        offscreen_window = glfw.create_window(self.width, self.height, "", None, None)
        glfw.make_context_current(offscreen_window)
        scene = mujoco.MjvScene(self._env.model, maxgeom=1000)
        context = mujoco.MjrContext(self._env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        while not self._terminated:
            if self._simulation.is_loop_state('pause'):
                continue

            mujoco.mjv_updateScene(self._env.model, self._env.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

            rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, self.width, self.height), scene, context)
            mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, self.width, self.height), context)
            rgb_image = np.flip(rgb_buffer, axis=0)
            self.frame_ready.emit(rgb_image)

            if self._predictor is not None:
                self.detection_ready.emit(self._predictor.predict(frame=rgb_image))

            time.sleep(1.0 / self._update_rate)
