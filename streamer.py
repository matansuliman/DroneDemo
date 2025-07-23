from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np
import mujoco
import glfw
import threading
import time

FPS = 20

WIDTH = 320
HEIGHT = 240

DISTANCE = 0
AZIMUTH = 0
ELEVATION = -90


class CameraStreamer(QObject):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, orchestrator, attached_body, update_rate =FPS):
        super().__init__()
        self.env = orchestrator._env
        self.attached_body = attached_body
        self.orchestrator = orchestrator
        self.update_rate = update_rate
        self.running = False

        self.width = WIDTH
        self.height = HEIGHT

        # Init visualization objects
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False

    def _run(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialized")
        
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # <- Hide the window
        offscreen_window = glfw.create_window(self.width, self.height, "", None, None)
        glfw.make_context_current(offscreen_window)

        scene = mujoco.MjvScene(self.env.model, maxgeom=1000)
        context = mujoco.MjrContext(self.env.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        while self.running:
            self.cam.lookat[:] = self.attached_body.sensors['gps'].getTruePos()
            self.cam.distance = DISTANCE
            self.cam.azimuth = AZIMUTH
            self.cam.elevation = ELEVATION

            if self.orchestrator.loop_paused:
                time.sleep(1.0 / self.update_rate)
                continue

            mujoco.mjv_updateScene(
                self.env.model, self.env.data, self.opt, None, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, scene
            )

            rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, self.width, self.height), scene, context)
            mujoco.mjr_readPixels(rgb_buffer, None,
                                  mujoco.MjrRect(0, 0, self.width, self.height),
                                  context)
            rgb_image = np.flip(rgb_buffer, axis=0)
            self.frame_ready.emit(rgb_image)

            time.sleep(1.0 / self.update_rate)
