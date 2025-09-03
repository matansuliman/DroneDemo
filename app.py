import sys
import threading
from PySide6.QtWidgets import QApplication

from simulation import SimulationRunner
from orchestrators import Follow
from guis import GUI
from streamers import CameraStreamer
from plots import plot_log

from logger_config import setup_logger


class App:
    def __init__(self, info: str= ''):
        logger.info("App: Initiating")
        self._info = info
        self._app = QApplication(sys.argv)
        self._simulation = SimulationRunner(orchestrator=Follow)
        self._camera_streamer = CameraStreamer(simulation=self._simulation)
        self._gui = GUI(self._simulation, self._camera_streamer)
        logger.info("App: Initiated")

    
    def run(self) -> None:
        logger.info("App: Running")

        # Connect camera streamer to gui functions
        self._camera_streamer.detection_ready.connect(self._gui.update_marker_detection)
        self._camera_streamer.frame_ready.connect(self._gui.update_camera_view)

        logger.debug("App: Start streaming in background thread")
        threading.Thread(target=self._camera_streamer.run, daemon=True).start()

        logger.debug("App: Start simulation loop in background thread")
        threading.Thread(target=self._simulation.run, daemon=True).start()

        logger.debug("App: Show gui in the main thread")
        self._gui.show()
        self._app.exec()


    def exit(self):
        logger.info("App: Exiting")
        plot_log(
            self._simulation.orchestrator.objects['quadrotor_controller'].log,
            self._simulation.orchestrator.objects['platform_controller'].log,
        )
        logger.debug("App: Plotted logs")
        logger.info("App: Exited")

if __name__ == "__main__":
    logger = setup_logger()
    myapp = App()
    myapp.run()
    myapp.exit()