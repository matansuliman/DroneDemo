import numpy as np

from models import Quadrotor, MovingPlatform
from controllers import QuadrotorController, MovingPlatformController
from environment import ENV
from predictors import MarkerDetector

import logging
logger = logging.getLogger("app")

CAMERA_DISTANCE_MULTIPLIER = 1.3
CAMERA_DISTANCE_OFFSET = 4

class BasicOrchestrator:
    def __init__(self, info: str= ''):
        self._info = info
        self._env = ENV()
        self._objects = dict() # {name: object}

    @property
    def info(self):
        return self._info

    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects
        
    def step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self):
        objs_str = "".join([f'\n\t\t{obj}' for name, obj in self._objects.items()])
        return f'orchestrator ({self.__class__.__name__}) info: {self._info}, objects:{objs_str}'

TOL_DISTANCE_FROM_PLATFORM = 0.02 # 2cm
PIXEL_TO_METER = 1/29 # 1px is 34.5cm

class Follow(BasicOrchestrator):
    def __init__(self, info: str= 'drone follow platform'):
        super().__init__(info=info)

        # Initialize objects
        quadrotor = Quadrotor(env=self._env)
        platform = MovingPlatform(env=self._env)
        self._objects = {
            'viewer': self._env.launch_viewer(),
            'quadrotor': quadrotor,
            'platform': platform,
            'quadrotor_controller': QuadrotorController(env=self._env, quadrotor=quadrotor),
            'platform_controller': MovingPlatformController(env=self._env, platform=platform)
        }

        # init wind
        self._env.enable_wind(True)
        self._env.set_wind(velocity_world=[0, 0, 0], air_density=1.225)

        # init params
        self._locked_rel_pos_xy = None
        self._predictor = MarkerDetector()
        self._predicted = False
        self._adjust = np.array([0,0,0])

        # init camera view
        self.update_camera_viewer()
    
    @property
    def predictor(self):
        return self._predictor

    @predictor.setter
    def predictor(self, predictor):
        self._predictor = predictor

    def update_camera_viewer(self):
        drone_pos = self._objects['quadrotor'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')
        avg_pos = np.average([drone_pos[:3], platform_pos[:3]], axis=0)
        self._objects['viewer'].cam.distance = CAMERA_DISTANCE_MULTIPLIER * avg_pos[2] + CAMERA_DISTANCE_OFFSET
        self._objects['viewer'].cam.lookat[:] = avg_pos

    def can_land(self):
        if not self.objects['quadrotor_controller'].descending and self._predictor.is_valid():
            last = np.array(self.predictor.get_last()) * PIXEL_TO_METER
            return np.linalg.norm(last) < self._objects["platform"].radius
        else:
            return False

    def _drone_is_close_to_platform(self):
        drone_pos = self._objects['quadrotor'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')

        if drone_pos[2] - platform_pos[2] < TOL_DISTANCE_FROM_PLATFORM:
            self._locked_rel_pos_xy = np.array([drone_pos[0] - platform_pos[0],
                                       drone_pos[1] - platform_pos[1]])
            return True
        return False

    def print_results(self, status: str= 'success'):
        acc = (1- self._locked_rel_pos_xy) * 100 / self._objects["platform"].radius
        logger.debug(f'status: {status} relpos: {self._locked_rel_pos_xy} , 'f'accuracy: {acc}%')
        
    def step_scene(self):
        # step platform
        self._objects['platform_controller'].step()
        
        # step drone
        if self._objects['platform_controller'].locks_activated:
            platform_pos = self._objects['platform'].get_pos(mode='no_noise')
            rel_pos = [self._locked_rel_pos_xy[0], self._locked_rel_pos_xy[1], 0]
            
            self._objects['quadrotor_controller'].update_target(mode = 'hardcode',
                                                                data = {'qpos' : platform_pos + rel_pos})

        elif self._drone_is_close_to_platform():
            self._objects['platform_controller'].activate_locks()
            self.print_results()
        
        else:
            if not self._predictor.predicted and self._predictor.is_valid() and self._predictor.is_stable():
                self._adjust = np.append(self._predictor.get_mean_from_history(), 0) * PIXEL_TO_METER
                self._predictor.predicted = True
                logger.debug(f"Orchestrator: adjusted target pos")

            self._objects['quadrotor_controller'].update_target(
                mode = 'follow',
                data = {'new_target_pos' : self._objects['platform'].get_pos(mode='noise') + self._adjust,
                        'new_target_vel' : self._objects['platform_controller'].velocity}
            )
            self._objects['quadrotor_controller'].step()

        # step camera view
        self.update_camera_viewer()

        # sync viewer
        self._objects['viewer'].sync()
