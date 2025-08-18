import time

import mujoco.viewer
import numpy as np

from models import Drone, MovingPlatform
from controllers import QuadrotorController, MovingPlatformController
from environment import ENV
from predictors import MarkerDetector


class BasicOrchestrator:
    def __init__(self):
        self._env = ENV()
        self._objects =  {} # {name: object} 

    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects
        
    def step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")

COOL_OFF_TIME = 10
DISTANCE_FROM_PLATFORM = 0.01
CAMERA_DISTANCE_MULTIPLIER = 1.3
CAMERA_DISTANCE_OFFSET = 4
PIXEL_TO_METER = 1/29

class FollowTarget(BasicOrchestrator):
    def init_env(self):
        self._env.enable_wind(True)
        self._env.set_wind(velocity_world=[0, 0, 0], air_density=1.225)

        # Quadratic drag coefficient*area (Cd*A) per body (rough guesses you can tune)
        # Larger value => more drag. Start small to avoid destabilizing.
        self._env.set_body_cda('x2', 0.04)
        self._env.set_body_cda('platform', 0.25)

    def init_object(self):
        # Initialize objects
        drone = Drone(self._env)
        platform = MovingPlatform(self._env)
        self._objects =  {
            'viewer': mujoco.viewer.launch_passive(self._env.model, self._env.data),
            'drone': drone,
            'platform': platform,
            'drone_controller': QuadrotorController(self._env, drone),
            'platform_controller': MovingPlatformController(self._env, platform)
        }

    def __init__(self):
        super().__init__()
        self.init_env()
        self.init_object()

        self._locked_rel_pos_xy = None
        self._predictor = MarkerDetector()
        self._adjust = np.array([0,0,0])
        self._target_adjusted = False
        self._start_time = time.time()
        
    
    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, predictor):
        self._predictor = predictor

    @property
    def target_adjusted(self):
        return self._target_adjusted

    def _drone_is_close_to_platform(self):
        drone_pos = self._objects['drone'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')

        if drone_pos[2] - platform_pos[2] < DISTANCE_FROM_PLATFORM:
            self._locked_rel_pos_xy = (drone_pos[0] - platform_pos[0],
                                       drone_pos[1] - platform_pos[1])
            return True
        return False
        
    def step_scene(self):
        # step platform
        self._objects['platform_controller'].step()
        
        # step drone
        if self._objects['platform_controller'].locks_activated:
            platform_pos = self._objects['platform'].get_pos(mode='no_noise')
            rel_pos = [self._locked_rel_pos_xy[0], self._locked_rel_pos_xy[1], DISTANCE_FROM_PLATFORM]
            
            self._objects['drone_controller'].update_target(mode = 'hardcode',
                                                            data = {'qpos' : platform_pos + rel_pos})

        elif self._drone_is_close_to_platform():
            self._objects['platform_controller'].activate_locks()
        
        else:
            if ((time.time() - self._start_time > COOL_OFF_TIME) and 
                not self._target_adjusted and
                self._predictor is not None and
                self._predictor.is_stable()):
                    
                    self._adjust = np.append(self._predictor.get_mean_from_history() * PIXEL_TO_METER, 0)
                    self._target_adjusted = True
            
            res = self._objects['platform'].get_pos(mode='noise') - self._adjust
            self._objects['drone_controller'].update_target(mode = 'follow',
                                                            data = {'new_target_pos' : res, 
                                                                    'new_target_vel' : self._objects['platform'].velocity}
                                                            )
            self._objects['drone_controller'].step()

        # step camera view
        drone_pos = self._objects['drone'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')

        self._objects['viewer'].cam.distance = CAMERA_DISTANCE_MULTIPLIER * abs(drone_pos[2] - platform_pos[2]) + CAMERA_DISTANCE_OFFSET
        self._objects['viewer'].cam.lookat[:] = (drone_pos[:3] + platform_pos[:3]) / 2

        # sync viewer
        self._objects['viewer'].sync()
