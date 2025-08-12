import sys
import threading
import glfw
from PyQt5.QtWidgets import QApplication

from simulation import SimulationRunner
from orchestrators import FollowTarget
from predictors import MarkerDetector
from guis import TargetControlGUI
from streamers import CameraStreamer
from plots import plot_log


if __name__ == "__main__":
    # Build orchestrator (scenario)
    orchestrator = FollowTarget()

    # Optional: Attach predictor
    detector = MarkerDetector()
    orchestrator.predictor = detector

    # Start simulation loop in background thread
    simulation = SimulationRunner(orchestrator)
    threading.Thread(target=simulation.run, daemon=True).start()

    # GUI + camera streamer
    app = QApplication(sys.argv)

    camera_streamer = CameraStreamer(
        simulation=simulation,
        attached_body=orchestrator.objects['drone'],
        predictor=detector,
    )

    gui = TargetControlGUI(simulation, camera_streamer)

    camera_streamer.detection_ready.connect(gui.update_marker_detection)
    camera_streamer.frame_ready.connect(gui.update_camera_view)
    camera_streamer.start()

    gui.show()
    app.exec_()

    # Plot after GUI closes
    plot_log(
        orchestrator.objects['drone_controller'].log,
        orchestrator.objects['platform_controller'].log,
    )
