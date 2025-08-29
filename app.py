import sys
import builtins
import datetime
import threading
from PySide6.QtWidgets import QApplication

# init logger file and print function
built_in_print = builtins.print
builtins.print = lambda data: built_in_print(f"[{datetime.datetime.now()}] {data}")
logger_file_descriptor = open("app.log", "w")
sys.stdout = logger_file_descriptor
sys.stderr = logger_file_descriptor

from simulation import SimulationRunner
from orchestrators import Follow
from guis import GUI
from streamers import CameraStreamer
from plots import plot_log


class App:
    def __init__(self, info: str= ''):
        self._info = info
        self._app = QApplication(sys.argv)
        self._simulation = SimulationRunner(orchestrator=Follow)
        self._camera_streamer = CameraStreamer(simulation=self._simulation, attached_body_name='quadrotor')
        self._gui = GUI(self._simulation, self._camera_streamer)




        print("App: Starting")
    
    def run(self) -> None:
        print("App: Running")
        # Connect camera streamer to gui functions and start streaming
        self._camera_streamer.detection_ready.connect(self._gui.update_marker_detection)
        self._camera_streamer.frame_ready.connect(self._gui.update_camera_view)
        self._camera_streamer.start()

        # Start simulation loop in background thread
        threading.Thread(target=self._simulation.run, daemon=True).start()

        # show gui in the main thread
        self._gui.show()
        self._app.exec()

    def exit(self):
        print("App: Plotting")
        plot_log(
            self._simulation.orchestrator.objects['quadrotor_controller'].log,
            self._simulation.orchestrator.objects['platform_controller'].log,
        )
        print("App: Exiting")

if __name__ == "__main__":
    myapp = App()
    myapp.run()
    myapp.exit()
