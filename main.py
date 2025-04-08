import sys
import threading

import mujoco
from PyQt5.QtWidgets import QApplication

from simulation import SimulationRunner
from drone_model import QuadrotorController
from platform_model import MovingPlatform
from streamer import CameraStreamer
from gui import TargetControlGUI

from plots import plot_log

PATH_TO_XML = "skydio_x2/scene.xml"

if __name__ == "__main__":

    import os
    print("cwd:", os.getcwd())
    print("Exists:", os.path.exists("skydio_x2/scene.xml"))
    print("Exists include:", os.path.exists("skydio_x2/x2.xml"))

    model = mujoco.MjModel.from_xml_path(PATH_TO_XML)
    data = mujoco.MjData(model)
    
    platform = MovingPlatform(model, data)
    drone = QuadrotorController(model, data)
    camera_streamer = CameraStreamer(model, data, 
                                     attached_body=drone)

    sim_thread = threading.Thread(
        target=lambda: SimulationRunner(model, data, drone, platform).run(),
        daemon=True
    )
    sim_thread.start()

    app = QApplication(sys.argv)
    gui = TargetControlGUI(drone, platform, camera_streamer)

    camera_streamer.frame_ready.connect(gui.update_camera_view)
    camera_streamer.start()

    gui.show()
    app.exec_()

    plot_log(drone.log, platform.log)
