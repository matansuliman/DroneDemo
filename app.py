import sys
import threading
from PySide6.QtWidgets import QApplication

from simulation import SimulationRunner
from orchestrators import Follow
from guis import GUI
from streamers import CameraStreamer

from logger import LOGGER


class App:
    def __init__(self):
        LOGGER.info("App: Initiating")
        self._app = QApplication(sys.argv)
        self._simulation = SimulationRunner(orchestrator=Follow)
        self._camera_streamer = CameraStreamer(simulation=self._simulation)
        self._gui = GUI(simulation=self._simulation)
        LOGGER.info("App: Initiated")

    def run(self) -> None:
        LOGGER.info("App: Running")
        LOGGER.debug("App: Connecting camera streamer to gui")
        self._camera_streamer.frame_ready.connect(
            self._gui.update_camera_streamer_frame
        )
        self._camera_streamer.status_ready.connect(
            self._gui.update_camera_streamer_status
        )
        LOGGER.debug("App: Connecting simulation to gui")
        self._simulation.status_ready.connect(self._gui.update_simulation_status)
        LOGGER.debug("App: Start streaming in background thread")
        threading.Thread(target=self._camera_streamer.run, daemon=True).start()
        LOGGER.debug("App: Start simulation loop in background thread")
        threading.Thread(target=self._simulation.run, daemon=True).start()
        LOGGER.debug("App: Show gui in the main thread")
        self._gui.show()
        self._app.exec()

    def exit(self):
        LOGGER.info("App: Exiting")
        self._simulation.plot_logs()
        LOGGER.info("App: Exited")