# file: dataset_logger.py
import sys
import os
import csv
import time
import threading
import numpy as np
import imageio
import mujoco
from PyQt5.QtWidgets import QApplication
from scipy.spatial.transform import Rotation as R

from models import QuadrotorController, MovingPlatform
from streamer import CameraStreamer
from gui import TargetControlGUI

class DatasetLogger:
    def __init__(self, out_dir="dataset", image_format="jpg"):
        self.out_dir = out_dir
        self.image_format = image_format
        self.csv_path = os.path.join(out_dir, "dataset.csv")
        self.counter = 0

        os.makedirs(out_dir, exist_ok=True)
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "image",
                "rel_x", "rel_y", "rel_z"
            ])

    def log(self, image, drone_pos, platform_pos):
        filename = f"frame_{self.counter:06d}.{self.image_format}"
        filepath = os.path.join(self.out_dir, filename)
        imageio.imwrite(filepath, image)

        def fmt(x):
            return f"{x:.5f}"

        rel_pos = platform_pos - drone_pos

        row = [
            filename,
            *[fmt(x) for x in rel_pos]
        ]

        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.counter += 1

def run_simulation(drone, platform, model, data, dt=0.01):
    while not drone.terminated:
        mujoco.mj_step(model, data)

        if not drone.paused:
            platform.step()
            #drone.update_target(platform)
            drone.step()

        time.sleep(dt)

if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path("skydio_x2/scene.xml")
    data = mujoco.MjData(model)

    platform = MovingPlatform(model, data)
    drone = QuadrotorController(model, data)
    camera_streamer = CameraStreamer(model, data, drone.body_id)

    # Right after creating the drone object
    drone.target[2] = 10.0
    drone.pid_z.setpoint = 10.0

    logger = DatasetLogger()
    camera_streamer.start()

    sim_thread = threading.Thread(target=run_simulation, args=(drone, platform, model, data), daemon=True)
    sim_thread.start()

    frame_tracker = {'count': 0}

    def update_logger():
        if frame_tracker['count'] % 5 == 0:
            pos = data.xpos[drone.body_id].copy()
            platform_pos = platform.get_position()
            if hasattr(camera_streamer, 'last_frame'):
                logger.log(camera_streamer.last_frame, pos, platform_pos)
        frame_tracker['count'] += 1

    app = QApplication(sys.argv)
    gui = TargetControlGUI(drone, platform, camera_streamer)

    def wrapped_update(frame):
        camera_streamer.last_frame = frame
        gui.update_camera_view(frame)
        update_logger()

    camera_streamer.frame_ready.connect(wrapped_update)

    gui.show()
    sys.exit(app.exec_())
