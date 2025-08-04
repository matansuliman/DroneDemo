import sys
import threading


from simulation import SimulationWrapper
from orchestrators import FollowTarget

from gui import TargetControlGUI
from streamer import CameraStreamer
from PyQt5.QtWidgets import QApplication

from plots import plot_log

def _start_simulation():
    sim_thread = threading.Thread(
        target=lambda: SimulationWrapper(orchestrator).run(),
        daemon=True
    )
    sim_thread.start()

def _start_gui():
    camera_streamer = CameraStreamer(orchestrator= orchestrator,
                                    attached_body= orchestrator._objects['drone'])

    app = QApplication(sys.argv)
    gui = TargetControlGUI(orchestrator, camera_streamer)

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
