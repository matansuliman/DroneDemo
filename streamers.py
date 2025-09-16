import glfw
import mujoco
import cv2
from PySide6.QtCore import Signal, QObject

from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG
from fps import BasicFPS


class CameraStreamer(QObject):
    frame_ready = Signal(np.ndarray)
    status_ready = Signal(str)

    def __init__(self, simulation):
        super().__init__()
        self._simulation = simulation

        self._fps = BasicFPS(frequency= CONFIG["camera_streamer"]["frequency"])

        self._resolutions = CONFIG["camera_streamer"]["resolutions"]

        # Init visualization objects
        self._option = mujoco.MjvOption()
        self._option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = mujoco.mj_name2id(
            ENVIRONMENT.model, mujoco.mjtObj.mjOBJ_CAMERA, "bottom_cam"
        )

        LOGGER.info(f"\tCameraStreamer: Initiated {self.__class__.__name__}")

    def status(self):
        status = f"{self.__class__.__name__} status:"
        status += f"\t\tcurrent fps: {print_for_gui(self._fps.curr_fps)}"
        status += f"\ttarget fps: {print_for_gui(self._fps.target_fps)}"
        return status

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not be initialized")

        w, h = self._resolutions["high"]
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.SAMPLES, 4)
        offscreen_window = glfw.create_window(w, h, "", None, None)
        glfw.make_context_current(offscreen_window)
        scene = mujoco.MjvScene(ENVIRONMENT.model, maxgeom=1000)
        context = mujoco.MjrContext(
            ENVIRONMENT.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        while self._simulation.continue_streaming():

            mujoco.mjv_updateScene(
                ENVIRONMENT.model,
                ENVIRONMENT.data,
                self._option,
                None,
                self._camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            rgb_buffer = np.zeros((h, w, 3), dtype=np.uint8)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), scene, context)
            mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, w, h), context)
            rgb_image = np.flip(rgb_buffer, axis=0)

            # stream to simulation
            self._simulation.stream(frame=rgb_image)  # stream frame to orchestrator

            # emit to gui
            gui_frame = cv2.resize(
                rgb_image, tuple(self._resolutions["low"]), interpolation=cv2.INTER_AREA
            )
            self.frame_ready.emit(gui_frame)  # emit frame to gui
            self.status_ready.emit(self.status())  # emit status to gui

            self._fps.maintain()
