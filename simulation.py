import time
import mujoco.viewer

from drone_model import QuadrotorController
from platform_model import MovingPlatform

DISTANCE_FROM_PLATFORM = 0.01
CAMERA_DISTANCE_MULTIPLIER = 1.5

CAMERA_DISTANCE_OFFSET = 3
PAUSE_SLEEP_SEC = 0.1

class SimulationRunner:

    def __init__(self,
                 model, 
                 data,
                 drone: QuadrotorController, 
                 platform: MovingPlatform,
                 dt: float = 0.01):
        
        self.model = model
        self.data = data
        self.dt = dt
    
        self.drone = drone
        self.platform = platform

        self.paused = False
        self.terminated = False

        self.rel_endx = None
        self.rel_endy = None

    def run(self):

        def _update_camera_view():
            viewer.cam.distance = CAMERA_DISTANCE_MULTIPLIER * abs(self.drone.gps.get_true_pos()[2] - self.platform.gps.get_true_pos()[2]) + CAMERA_DISTANCE_OFFSET
            viewer.cam.lookat[:] = (self.drone.gps.get_true_pos()[:3] + self.platform.gps.get_true_pos()[:3]) / 2

        def _drone_is_close_to_platform():
            if self.drone.gps.get_true_pos()[2] - self.platform.gps.get_true_pos()[2] < DISTANCE_FROM_PLATFORM:
                self.rel_endx = self.drone.gps.get_true_pos()[0] - self.platform.gps.get_true_pos()[0]
                self.rel_endy = self.drone.gps.get_true_pos()[1] - self.platform.gps.get_true_pos()[1]
                return True
            return False 

        def _lock_drone_to_platform():
            self.drone.data.qpos[:3] = self.platform.gps.get_true_pos()[:3] + [self.rel_endx, self.rel_endy, DISTANCE_FROM_PLATFORM]
            self.drone.data.qvel[:] = 0
            self.drone.data.qacc[:] = 0
            self.drone.data.qpos[3:7] = [1, 0, 0, 0]  # upright orientation

        with mujoco.viewer.launch_passive(self.drone.model, self.drone.data) as viewer:
            while True:
                mujoco.mj_step(self.model, self.data)

                if self.drone.terminated:
                    break

                if self.drone.paused:
                    time.sleep(PAUSE_SLEEP_SEC)
                    continue
                
                self.platform.step()

                if not self.drone.lock and _drone_is_close_to_platform():
                    self.drone.lock = True
                
                if self.drone.lock:
                    _lock_drone_to_platform()
                else:
                    self.drone.update_target(self.platform.gps.get_true_pos(), self.platform.get_vel()[:2])
                    self.drone.step()

                _update_camera_view()

                viewer.sync()
                time.sleep(self.dt)

