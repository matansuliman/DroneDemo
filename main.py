import sys
import threading
import mujoco

from environment import ENV
from simulation import SimulationRunner

from models import Drone, MovingPlatform
from controllers import QuadrotorController, MovingPlatformController
from orchestrator import Orchestrator

from gui import TargetControlGUI
from streamer import CameraStreamer
from PyQt5.QtWidgets import QApplication

from plots import plot_log

PATH_TO_XML = "skydio_x2/scene.xml"

def init_env():
    model = mujoco.MjModel.from_xml_path(PATH_TO_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    return ENV(model, data, dt)

def init_objects(env):

    drone = Drone(env)
    platform = MovingPlatform(env)
    drone_controller = QuadrotorController(env, drone)
    platform_controller = MovingPlatformController(env, platform)

    return {
        'drone': drone,
        'platform': platform,
        'drone_controller': drone_controller,
        'platform_controller': platform_controller,
    }

def init_orchestrator():
    env = init_env()
    return Orchestrator(
        env=env,
        objects=init_objects(env)
    )

if __name__ == "__main__":

    orchestrator = init_orchestrator()

    sim_thread = threading.Thread(
        target=lambda: SimulationRunner(orchestrator).run(),
        daemon=True
    )
    sim_thread.start()

    camera_streamer = CameraStreamer(orchestrator= orchestrator,
                                     attached_body= orchestrator._objects['drone'])

    app = QApplication(sys.argv)
    gui = TargetControlGUI(orchestrator, camera_streamer)

    camera_streamer.frame_ready.connect(gui.update_camera_view)
    camera_streamer.start()

    gui.show()
    app.exec_()

    plot_log(orchestrator._objects['drone_controller'].log, orchestrator._objects['platform_controller'].log)
