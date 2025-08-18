import sys
import threading
from PySide6.QtWidgets import QApplication

from simulation import SimulationRunner
from orchestrators import FollowTarget
from guis import TargetControlGUI
from streamers import CameraStreamer
from plots import plot_log

class App:
    def __init__(self, args=None):
        if args is None:
            args = {}
        self._args = args
    
    def start(self) -> None:

        print(self._args)

        # Start simulation loop in background thread
        simulation = SimulationRunner(FollowTarget)
        threading.Thread(target=simulation.run, daemon=True).start()

        # GUI + camera streamer
        app = QApplication(sys.argv)

        camera_streamer = CameraStreamer(
            simulation=simulation,
            attached_body=simulation.orchestrator.objects['drone'],
            predictor=simulation.orchestrator.predictor,
        )

        gui = TargetControlGUI(simulation, camera_streamer)

        camera_streamer.detection_ready.connect(gui.update_marker_detection)
        camera_streamer.frame_ready.connect(gui.update_camera_view)
        camera_streamer.start()

        gui.show()
        app.exec()

        # Plot after GUI closes
        plot_log(
            simulation.orchestrator.objects['drone_controller'].log,
            simulation.orchestrator.objects['platform_controller'].log,
        )


if __name__ == "__main__":
    print("Starting Drone App")
    myapp = App()
    myapp.start()
    print("Exiting Drone App")
