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

        self._viewer = ENVIRONMENT.launch_viewer()

        # Initialize objects
        quadrotor = Quadrotor()
        pad = Pad()
        self._objects = {
            'Quadrotor': quadrotor,
            'Pad': pad,
            'Quadrotor_controller': QuadrotorController(quadrotor= quadrotor),
            'Pad_controller': PadController(pad= pad)
        }

        self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        # Initialize params
        self._scene_ended = False
        self._predictor = ArUcoMarkerPredictor()

        # Initialize wind
        config_env = CONFIG["Follow_Orch"]["env"]
        ENVIRONMENT.enable_wind(True)
        ENVIRONMENT.set_wind(velocity_world= config_env["default_wind"], air_density= config_env["air_density"])

        # Initialize camera view
        self._update_viewer_camera()

        LOGGER.info(f"\t\tOrchestrator: Initiated {self.__class__.__name__}")
    
    @property
    def predictor(self):
        return self._predictor

    @property
    def scene_ended(self):
        return self._scene_ended

    def _update_viewer_camera(self):
        drone_pos = self._objects['Quadrotor'].get_true_pos()
        pad_pos = self._objects['Pad'].get_true_pos()
        avg_pos = np.average([drone_pos, pad_pos], axis=0)

        self._viewer.cam.distance = CONFIG["Follow_Orch"]["viewer"]["camera_distance_coef"] * avg_pos[2] + CONFIG["Follow_Orch"]["viewer"]["camera_distance_ff"]
        self._viewer.cam.lookat[:] = avg_pos
        self._viewer.cam.azimuth = 90
        self._viewer.cam.elevation = -45

    def _drone_above_pad(self):
        q, p, _, _ = self._objects.values()
        curr_height = q.get_height()
        norm_from_center = np.linalg.norm(self._predictor.get_last_from_model())
        return norm_from_center < p.radius

    def _can_land(self):
        stable_short_term = self._predictor.is_model_stable(mode= 'short-term')
        return stable_short_term and self._drone_above_pad()

    def status(self):
        status = "" #f"{self.__class__.__name__} status:\n"
        status += self._predictor.status()
        for obj in self._objects.values(): status += obj.status()
        #status += f"{self._predictor.is_model_stable(mode= 'short-term')}\n"
        #status += f"{self._drone_above_pad()}\n"
        return status

    def stream(self, frame):
        curr_height = self._objects['Quadrotor'].get_height()
        self._predictor.stream_to_model(frame= frame, curr_height= curr_height)

    def _step_predictor(self):
        #if not qc.descend: self._predictor.predict(curr_height= q.get_height())
        curr_height = self._objects['Quadrotor'].get_height()
        self._predictor.predict()

    def _step_pad(self):
        self._objects['Pad_controller'].step()

    def _step_drone(self):
        _, p, qc, pc = self._objects.values()

        qc.descend = self._can_land()
        new_ref_pos, new_ref_vel = p.get_pos(), pc.velocity

        new_ref_pos += self._predictor.prediction  # use predicator
        qc.set_reference(pos=new_ref_pos, vel=new_ref_vel)
        qc.step()

    def _step_viewer(self):
        self._update_viewer_camera()  # step camera view
        self._viewer.sync()  # sync viewer

    def step_scene(self):
        self._step_predictor()
        self._step_pad()
        self._step_drone()
        self._step_viewer()