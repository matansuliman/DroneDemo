import sys
import threading
from PySide6.QtWidgets import QApplication

from simulation import SimulationRunner
from orchestrators import Follow
from guis import GUI
from streamers import CameraStreamer
from plots import plot_log

from logger import LOGGER
from config import CONFIG


class App:
    def __init__(self):
        LOGGER.info("App: Initiating")
        self._app = QApplication(sys.argv)
        self._simulation = SimulationRunner(orchestrator=Follow)
        self._camera_streamer = CameraStreamer(simulation=self._simulation)
        self._gui = GUI(simulation= self._simulation, camera_streamer=self._camera_streamer)
        LOGGER.info("App: Initiated")

    
    def run(self) -> None:
        LOGGER.info("App: Running")
        LOGGER.debug("App: Connecting camera streamer to gui functions")
        self._camera_streamer.detection_ready.connect(self._gui.update_marker_detection)
        self._camera_streamer.frame_ready.connect(self._gui.update_camera_view)
        LOGGER.debug("App: Start streaming in background thread")
        threading.Thread(target=self._camera_streamer.run, daemon=True).start()
        LOGGER.debug("App: Start simulation loop in background thread")
        threading.Thread(target=self._simulation.run, daemon=True).start()
        LOGGER.debug("App: Show gui in the main thread")
        self._gui.show()
        self._app.exec()

    def exit(self):
        LOGGER.info("App: Exiting")
        drone_log = self._simulation.orchestrator.objects['quadrotor_controller'].log
        platform_log = self._simulation.orchestrator.objects['platform_controller'].log
        plot_log(drone_log, platform_log)
        LOGGER.info("App: Exited")

if __name__ == "__main__":
    myapp = App()
    myapp.run()
    myapp.exit()