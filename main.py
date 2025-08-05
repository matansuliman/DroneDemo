import sys
import threading
from PyQt5.QtWidgets import QApplication

from simulation import SimulationWrapper
from orchestrators import FollowTarget
from predictors import MarkerDetector
from guis import TargetControlGUI
from streamers import CameraStreamer


from plots import plot_log

def _start_simulation():
    sim_thread = threading.Thread(
        target=lambda: SimulationWrapper(orchestrator).run(),
        daemon=True
    )
    sim_thread.start()

def _start_gui():
    detector = MarkerDetector()
    orchestrator._predictor=detector
    camera_streamer = CameraStreamer(orchestrator= orchestrator,
                                    attached_body= orchestrator._objects['drone'],
                                    predictor=detector)

    app = QApplication(sys.argv)
    gui = TargetControlGUI(orchestrator, camera_streamer)

    camera_streamer.detection_ready.connect(gui.update_marker_detection)
    camera_streamer.frame_ready.connect(gui.update_camera_view)
    camera_streamer.start()

    gui.show()
    app.exec_()

def _plot():
    plot_log(orchestrator._objects['drone_controller'].log, orchestrator._objects['platform_controller'].log)

if __name__ == "__main__":
    orchestrator = FollowTarget()
    _start_simulation()
    _start_gui()
    _plot()
