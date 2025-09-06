import mujoco
import numpy as np

from models import Quadrotor, Pad
from controllers import QuadrotorController, PadController
from predictors import ArUcoMarkerPredictor

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicOrchestrator:
    def __init__(self):
        LOGGER.info("\t\tOrchestrator: Initiating")
        self._objects = dict() # {name: object}
    
    @property
    def objects(self):
        return self._objects

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")
        
    def step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")


class Follow(BasicOrchestrator):
    def __init__(self):
        super().__init__()

        # Initialize objects
        quadrotor = Quadrotor()
        pad = Pad()
        self._objects = {
            'viewer': ENVIRONMENT.launch_viewer(),
            'Quadrotor': quadrotor,
            'Pad': pad,
            'Quadrotor_controller': QuadrotorController(quadrotor= quadrotor),
            'Pad_controller': PadController(pad= pad)
        }

        v = self._objects['viewer']
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        # Initialize params
        self._scene_ended = False
        self._predictor = ArUcoMarkerPredictor()

        # Initialize wind
        config_env = CONFIG["Follow_Orch"]["env"]
        ENVIRONMENT.enable_wind(True)
        ENVIRONMENT.set_wind(velocity_world= config_env["default_wind"], air_density= config_env["air_density"])

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
        drone_pos = self._objects['Quadrotor'].get_true_pos()
        pad_pos = self._objects['Pad'].get_true_pos()
        avg_pos = np.average([drone_pos[:3], pad_pos[:3]], axis=0)
        #self._objects['viewer'].cam.distance = CONFIG["Follow_Orch"]["viewer"]["camera_distance_coef"] * avg_pos[2] + CONFIG["Follow_Orch"]["viewer"]["camera_distance_ff"]
        self._objects['viewer'].cam.lookat[:] = drone_pos # avg_pos

    def can_land(self):
        if self.objects['Quadrotor_controller'].descending: return False
        if not self._predictor.model.is_stable(mode= 'short-term'): return False
        return np.linalg.norm(self._predictor.model.get_last()) < self._objects["Pad"].radius

    def _pad_can_catch_drone(self):
        return self._objects['Quadrotor'].sensors['rangefinder'].get() < self._objects['Pad'].locks_arms_length

    def _results(self, status: str= 'SUCCESS'):
        abs_dis_from_center = abs(self._objects['Pad'].locks_end_pos)
        radius = self._objects["Pad"].radius
        precent = lambda whole, part: 100 * (1 - part/whole)
        accuracy = precent(radius, abs_dis_from_center)
        return f" *** {status} ***\tAccuracy[x,y]: {accuracy}%"

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += self._predictor.status()
        status += self._objects['Quadrotor_controller'].status()
        status += self._objects['Quadrotor'].status()
        status += self._objects['Pad_controller'].status()
        status += self._objects['Pad'].status()
        if self._scene_ended: status += f"results: {self._results()}"
        return status
        
    def step_scene(self):
        self._objects['Pad_controller'].step() # step pad
        
        """# step drone
        if self._objects['Pad_controller'].locks_activated:
            pad_pos = self._objects['Pad'].get_true_pos()
            rel_pos = np.append(self._objects['Pad'].locks_end_pos, 0)
            self._objects['Quadrotor_controller'].teleport(pos= pad_pos + rel_pos)
            self._scene_ended = True

        elif self._pad_can_catch_drone():
            drone_pos = self._objects['Quadrotor'].get_true_pos()
            pad_pos = self._objects['Pad'].get_true_pos()
            rel_pos = drone_pos - pad_pos
            self._objects['Pad_controller'].activate_locks(pos= rel_pos[:2])

        else:"""
        new_reference_pos = self._objects['Pad'].get_pos()
        new_reference_vel = self._objects['Pad_controller'].velocity
        self._predictor.predict()
        new_reference_pos += self._predictor.prediction # use predicator
        self._objects['Quadrotor_controller'].set_reference(pos= new_reference_pos, vel= new_reference_vel)
        self._objects['Quadrotor_controller'].step()

        self._update_camera_viewer() # step camera view
        self._objects['viewer'].sync() # sync viewer
