import numpy as np

from environment import ENV
from models import Quadrotor, MovingPlatform
from controllers import QuadrotorController, MovingPlatformController
from predictors import ArUcoMarkerPredictor

from logger import LOGGER
from config import CONFIG


class BasicOrchestrator:
    def __init__(self):
        LOGGER.info("\t\tOrchestrator: Initiating")
        self._env = ENV()
        self._objects = dict() # {name: object}

    @property
    def env(self):
        return self._env
    
    @property
    def objects(self):
        return self._objects
        
    def step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")


class Follow(BasicOrchestrator):
    def __init__(self):
        super().__init__()

        # Initialize objects
        quadrotor = Quadrotor(env= self._env)
        platform = MovingPlatform(env= self._env)
        self._objects = {
            'viewer': self._env.launch_viewer(),
            'quadrotor': quadrotor,
            'platform': platform,
            'quadrotor_controller': QuadrotorController(env= self._env, quadrotor= quadrotor),
            'platform_controller': MovingPlatformController(env= self._env, platform= platform)
        }

        # Initialize params
        self._scene_ended = False
        self._predictor = ArUcoMarkerPredictor()

        # Initialize wind
        config_env = CONFIG["Follow_Orch"]["env"]
        self._env.enable_wind(True)
        self._env.set_wind(velocity_world= config_env["default_wind"], air_density= config_env["air_density"])

        # Initialize camera view
        self._update_camera_viewer()

        LOGGER.info(f"\t\tOrchestrator: Initiated {self.__class__.__name__}")
    
    @property
    def predictor(self):
        return self._predictor

    @property
    def scene_ended(self):
        return self._scene_ended

    @predictor.setter
    def predictor(self, predictor):
        self._predictor = predictor

    def _update_camera_viewer(self):
        drone_pos = self._objects['quadrotor'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')
        avg_pos = np.average([drone_pos[:3], platform_pos[:3]], axis=0)
        self._objects['viewer'].cam.distance = CONFIG["Follow_Orch"]["viewer"]["camera_distance_coef"] * avg_pos[2] + CONFIG["Follow_Orch"]["viewer"]["camera_distance_ff"]
        self._objects['viewer'].cam.lookat[:] = avg_pos

    def can_land(self):
        #print(not self.objects['quadrotor_controller'].descending, self._predictor.predicted)
        if not self.objects['quadrotor_controller'].descending and self._predictor.predicted:
            return np.linalg.norm(self._predictor.get_last()) < self._objects["platform"].radius
        else:
            return False

    def _drone_is_close_to_platform(self):
        drone_pos = self._objects['quadrotor'].get_pos(mode='no_noise')
        platform_pos = self._objects['platform'].get_pos(mode='no_noise')

        if drone_pos[2] - platform_pos[2] < self._objects['platform'].locks_arms_length:
            self._objects['platform'].locks_end_pos = np.array([drone_pos[0] - platform_pos[0],
                                                                drone_pos[1] - platform_pos[1]])
            return True
        return False

    def print_results(self, status: str= 'SUCCESS'):
        abs_dis_from_center = abs(self._objects['platform'].locks_end_pos)
        radius = self._objects["platform"].radius
        precent = lambda whole, part: 100 * (1 - part/whole)
        accuracy = precent(radius, abs_dis_from_center)
        LOGGER.debug(f" *** {status} ***\tAccuracy[x,y]: {accuracy}%")
        
    def step_scene(self):
        self._objects['platform_controller'].step() # step platform
        
        # step drone
        if self._objects['platform_controller'].locks_activated:
            platform_pos = self._objects['platform'].get_pos(mode='no_noise')
            rel_pos = np.append(self._objects['platform'].locks_end_pos, 0)
            self._objects['quadrotor_controller'].set_reference(pos= platform_pos + rel_pos)
            self.print_results()
            self._scene_ended = True

        elif self._drone_is_close_to_platform():
            self._objects['platform_controller'].activate_locks()

        else:
            new_target_pos = self._objects['platform'].get_pos(mode='noise')
            if self.predictor.predicted: new_target_pos += self._predictor.prediction # use predicator
            self._objects['quadrotor_controller'].set_reference(pos= new_target_pos, vel= self._objects['platform_controller'].velocity)
            self._objects['quadrotor_controller'].step()

        self._update_camera_viewer() # step camera view
        self._objects['viewer'].sync() # sync viewer
