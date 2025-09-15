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
        self._viewer = ENVIRONMENT.launch_viewer()
        self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self._models = dict()
        self._controllers = dict()

    @property
    def models(self):
        return self._models

    @property
    def controllers(self):
        return self._controllers

    def get_logs(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _update_logs(self):
        raise NotImplementedError("Subclasses should implement this method")

    def set_pad_vel(self, vel):
        raise NotImplementedError("Subclasses should implement this method")

    def get_pad_vel(self):
        raise NotImplementedError("Subclasses should implement this method")

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def step_scene(self):
        raise NotImplementedError("Subclasses should implement this method")


class Follow(BasicOrchestrator):
    def __init__(self):
        super().__init__()

        self._models = {
            "Quadrotor": Quadrotor(),
            "Pad": Pad(),
        }

        self._controllers = {
            "Quadrotor": QuadrotorController(quadrotor=self._models["Quadrotor"]),
            "Pad": PadController(pad=self._models["Pad"]),
        }

        self._predictor = ArUcoMarkerPredictor()

        # init wind
        config_env = CONFIG["Follow_Orch"]["env"]
        ENVIRONMENT.enable_wind(True)
        ENVIRONMENT.set_wind(
            velocity_world=config_env["default_wind"],
            air_density=config_env["air_density"],
        )

        # initialize camera view
        self._update_viewer_camera()

        LOGGER.info(f"\t\tOrchestrator: Initiated {self.__class__.__name__}")

    def scene_ended(self):
        return self._controllers["Quadrotor"].is_done()

    # Camera Streamer helpers through simulation
    def stream(self, frame):
        curr_height = self._models["Quadrotor"].get_height()
        self._predictor.stream_to_model(frame=frame, curr_height=curr_height)

    # simulation helpers
    def get_logs(self):
        logs_dict = dict()

        models_logs = {name: model.log for name, model in self._models.items()}
        logs_dict.update(models_logs)

        # controllers_logs = {name: controller.log for name, controller in self._controllers.items()}
        # logs_dict.update(controllers_logs)

        return logs_dict

    # GUI helpers through simulation
    def status(self):
        status = ""  # f"{self.__class__.__name__} status:\n"
        status += self._predictor.status()
        for model in self._models.values():
            status += model.status()
        for controller in self._controllers.values():
            status += controller.status()
        return status

    def set_pad_vel(self, vel):
        self._controllers["Pad"].velocity = vel

    def get_pad_vel(self):
        return self._models["Pad"].get_vel()

    # step_scene helpers
    def _update_viewer_camera(self):
        drone_pos = self._models["Quadrotor"].get_true_pos()
        pad_pos = self._models["Pad"].get_true_pos()
        avg_pos = np.average([drone_pos, pad_pos], axis=0)
        diff_pos = drone_pos - pad_pos

        self._viewer.cam.distance = (
            CONFIG["Follow_Orch"]["viewer"]["camera_distance_coef"] * avg_pos[2]
            + CONFIG["Follow_Orch"]["viewer"]["camera_distance_ff"]
        )
        self._viewer.cam.lookat[:] = avg_pos
        self._viewer.cam.azimuth = 90
        self._viewer.cam.elevation = np.clip(
            -45 * np.linalg.norm(diff_pos) / 10, -45, -15
        )

    def _drone_above_pad(self):
        _, p = self._models.values()
        norm_from_center = np.linalg.norm(self._predictor.get_last_from_model())
        epsilon = 0.2
        return norm_from_center + epsilon < p.radius

    def _can_land(self):
        stable_short_term = self._predictor.is_model_stable(mode="short-term")
        return stable_short_term and self._drone_above_pad()

    def _step_predictor(self):
        self._predictor.predict()

    def _step_pad(self):
        self._controllers["Pad"].step()

    def _step_drone(self):
        (
            _,
            p,
        ) = self.models.values()
        qc, pc = self.controllers.values()

        qc.descend = self._can_land()
        new_ref_pos, new_ref_vel = p.get_pos(), p.get_vel()

        new_ref_pos += self._predictor.prediction  # use predicator
        qc.set_reference(pos=new_ref_pos, vel=new_ref_vel)
        qc.step()

    def _step_viewer(self):
        self._update_viewer_camera()  # step camera view
        self._viewer.sync()  # sync viewer

    def _update_logs(self):
        for model in self._models.values():
            model.update_log()
        # for controller in controllers.values(): controller.update_log()

    def step_scene(self):
        self._step_predictor()
        self._step_pad()
        self._step_drone()
        self._step_viewer()

        if not self.scene_ended():
            self._update_logs()
