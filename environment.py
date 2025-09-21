from __future__ import annotations
from dataclasses import dataclass, field

from typing import Dict, Iterable, Optional, Union
import mujoco
import mujoco.viewer
import glfw
import numpy as np

from config import CONFIG


@dataclass
class WindConfig:
    enabled: bool = False
    velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # m/s, world frame
    air_density: float = 1.225  # kg/m^3
    # Per-body drag coefficient * reference area (Cd * A) [m^2]
    # e.g., {"x2": 0.05, "platform": 0.2}
    cda_by_body: Dict[str, float] = field(default_factory=dict)


class ENV:
    def __init__(self, path_to_xml):
        self._model = mujoco.MjModel.from_xml_path(path_to_xml)
        self._data = mujoco.MjData(self._model)
        self._dt = self._model.opt.timestep

        # --- Extras state ---
        self._wind = WindConfig()

        # Fast name->id caches (created lazily)
        self._body_id_cache: Dict[str, int] = {}
        self._joint_id_cache: Dict[str, int] = {}
        self._act_id_cache: Dict[str, int] = {}
        self._sensor_id_cache: Dict[str, int] = {}

    def launch_viewer(self):
        glfw.init()
        return mujoco.viewer.launch_passive(self._model, self._data)

    # -------------------- Core properties --------------------
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def dt(self) -> float:
        return self._dt

    def get_time(self) -> float:
        return self._data.time

    # -------------------- Simulation stepping --------------------
    def step(self, n_substeps: int = 1):
        """
        Advance simulation by n_substeps * model.opt.timestep while applying configured effects.
        Replace direct calls to mujoco.mj_step(...) with env.step() to enable wind/drag, etc.
        """
        for _ in range(n_substeps):
            # Clear any previously applied external forces that *we* manage
            self.clear_external_forces()

            # Inject wind/drag if enabled
            if self._wind.enabled:
                self._apply_wind_drag()

            mujoco.mj_step(self._model, self._data)

    # -------------------- Wind / aerodynamic drag --------------------
    def enable_wind(self, enabled: bool = True):
        self._wind.enabled = enabled

    def set_wind(
        self,
        velocity_world: Union[Iterable[float], np.ndarray],
        air_density: Optional[float] = None,
    ):
        self._wind.velocity_world = np.asarray(velocity_world, dtype=float).reshape(3)
        if air_density is not None:
            self._wind.air_density = float(air_density)

    def get_wind_vel(self):
        return self._wind.velocity_world

    def set_body_cda(self, body: Union[str, int], cda: float):
        """Set Cd*A (m^2) used for quadratic drag on a body."""
        name = self.body_name(body)
        self._wind.cda_by_body[name] = float(cda)

    def clear_cda(self, body: Optional[Union[str, int]] = None):
        if body is None:
            self._wind.cda_by_body.clear()
        else:
            name = self.body_name(body)
            self._wind.cda_by_body.pop(name, None)

    def _apply_wind_drag(self):
        """
        Apply simple quadratic drag to configured bodies:
            F = 0.5 * rho * CdA * |v_rel| * v_rel  (opposes body motion relative to wind)
        Forces are applied at body COM in world frame (xfrc_applied).
        """
        rho = self._wind.air_density
        v_w = self._wind.velocity_world

        if not self._wind.cda_by_body:
            return  # nothing to do

        # data.cvel: (nbody, 6), spatial velocity in *body* frame [ang(3), lin(3)]
        # Convert its linear part to world frame using body xmat.
        cvel = self._data.cvel  # shape (nbody, 6)
        xmat = self._data.xmat.reshape(self._model.nbody, 3, 3)

        for body_name, cda in self._wind.cda_by_body.items():
            bid = self.body_id(body_name)

            v_lin_body = cvel[bid, 3:]  # linear velocity in body frame
            r_bw = xmat[bid]  # body->world
            v_lin_world = r_bw @ v_lin_body

            # relative wind: air minus body velocity (world frame)
            v_rel = v_w - v_lin_world
            speed = np.linalg.norm(v_rel)
            if speed <= 1e-9 or cda <= 0.0 or rho <= 0.0:
                continue

            # Quadratic drag, direction opposing relative motion
            f = 0.5 * rho * cda * speed * v_rel  # world frame

            # Apply at COM (no torque)
            self._data.xfrc_applied[bid, :3] += f
            # torque left zero

    # -------------------- External forces helpers --------------------
    def add_external_force(
        self,
        body: Union[str, int],
        force_world: Union[Iterable[float], np.ndarray],
        torque_world: Union[Iterable[float], np.ndarray] = (0.0, 0.0, 0.0),
        mode: str = "add",
    ):
        """
        Apply an external spatial force (world frame) on a body COM for the current step.
        If mode == 'set' it overwrites; if 'add' it accumulates.
        """
        bid = self.body_id(body)
        f = np.asarray(force_world, dtype=float).reshape(3)
        t = np.asarray(torque_world, dtype=float).reshape(3)

        if mode == "set":
            self._data.xfrc_applied[bid, :3] = f
            self._data.xfrc_applied[bid, 3:] = t
        elif mode == "add":
            self._data.xfrc_applied[bid, :3] += f
            self._data.xfrc_applied[bid, 3:] += t
        else:
            raise ValueError("mode must be 'set' or 'add'")

    def clear_external_forces(self):
        """Zero out all xfrc_applied for this step (safe to call every step)."""
        self._data.xfrc_applied[:, :] = 0.0

    # -------------------- World/options setters --------------------
    def set_gravity(self, g_world: Union[Iterable[float], np.ndarray]):
        self._model.opt.gravity[:] = np.asarray(g_world, dtype=float).reshape(3)

    def set_air_density(self, rho: float):
        self._model.opt.density = float(rho)

    def set_viscosity(self, mu: float):
        self._model.opt.viscosity = float(mu)

    def set_timestep(self, dt: float):
        self._model.opt.timestep = float(dt)
        self._dt = float(dt)

    # -------------------- Reset helpers --------------------
    def reset(
        self,
        qpos: Optional[Iterable[float]] = None,
        qvel: Optional[Iterable[float]] = None,
    ):
        """mj_resetData + optional state overwrite."""
        mujoco.mj_resetData(self._model, self._data)
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=float)
            if qpos.size != self._model.nq:
                raise ValueError(
                    f"qpos has size {qpos.size}, expected {self._model.nq}"
                )
            self._data.qpos[:] = qpos
        if qvel is not None:
            qvel = np.asarray(qvel, dtype=float)
            if qvel.size != self._model.nv:
                raise ValueError(
                    f"qvel has size {qvel.size}, expected {self._model.nv}"
                )
            self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)

    # -------------------- Body / joint utilities --------------------
    def body_id(self, body: Union[str, int]) -> int:
        if isinstance(body, int):
            return int(body)
        if body in self._body_id_cache:
            return self._body_id_cache[body]
        bid = self._model.body(body).id
        self._body_id_cache[body] = bid
        return bid

    def body_name(self, body: Union[str, int]) -> str:
        if isinstance(body, str):
            return body
        return self._model.body(body).name

    def joint_id(self, joint: Union[str, int]) -> int:
        if isinstance(joint, int):
            return int(joint)
        if joint in self._joint_id_cache:
            return self._joint_id_cache[joint]
        jid = self._model.joint(joint).id
        self._joint_id_cache[joint] = jid
        return jid

    def actuator_id(self, actuator: Union[str, int]) -> int:
        if isinstance(actuator, int):
            return int(actuator)
        if actuator in self._act_id_cache:
            return self._act_id_cache[actuator]
        aid = self._model.actuator(actuator).id
        self._act_id_cache[actuator] = aid
        return aid

    def sensor_id(self, sensor: Union[str, int]) -> int:
        """
        Get a sensor id from name or pass-through if already an int.
        Caches lookups by name for speed.
        """
        if isinstance(sensor, int):
            return int(sensor)
        if sensor in self._sensor_id_cache:
            return self._sensor_id_cache[sensor]
        sid = self._model.sensor(sensor).id
        self._sensor_id_cache[sensor] = sid
        return sid

    def sensor_name(self, sensor: Union[str, int]) -> str:
        """Return sensor name given id or pass-through if already a name."""
        if isinstance(sensor, str):
            return sensor
        return self._model.sensor(sensor).name

    # ---- Poses for free bodies and joints ----
    def set_free_body_pose(
        self,
        body: Union[str, int],
        pos_world: Optional[Iterable[float]] = None,
        quat_wxyz: Optional[Iterable[float]] = None,
    ):
        """
        Set the pose for a body with a freejoint. Quat must be [w,x,y,z].
        """
        bid = self.body_id(body)
        # Get the freejoint that belongs to this body
        jid = self._model.body_jntnum[bid]
        if jid <= 0:
            raise ValueError(
                f"Body '{self.body_name(bid)}' has no joint or not a freejoint."
            )
        jadr = self._model.jnt_qposadr[self._model.body_jntadr[bid]]
        jtype = self._model.jnt_type[self._model.body_jntadr[bid]]
        if jtype != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"Body '{self.body_name(bid)}' primary joint is not free (type={int(jtype)})."
            )

        if pos_world is not None:
            self._data.qpos[jadr : jadr + 3] = np.asarray(
                pos_world, dtype=float
            ).reshape(3)
        if quat_wxyz is not None:
            self._data.qpos[jadr + 3 : jadr + 7] = np.asarray(
                quat_wxyz, dtype=float
            ).reshape(4)

    def set_free_body_velocity(
        self,
        body: Union[str, int],
        linvel_world: Optional[Iterable[float]] = None,
        angvel_world: Optional[Iterable[float]] = None,
    ):
        """
        Set the spatial velocity for a freejoint body (qvel portion).
        MuJoCo freejoint velocity order: [3 ang, 3 lin].
        """
        bid = self.body_id(body)
        jid = self._model.body_jntnum[bid]
        if jid <= 0:
            raise ValueError(
                f"Body '{self.body_name(bid)}' has no joint or not a freejoint."
            )
        vadr = self._model.jnt_dofadr[self._model.body_jntadr[bid]]
        jtype = self._model.jnt_type[self._model.body_jntadr[bid]]
        if jtype != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"Body '{self.body_name(bid)}' primary joint is not free (type={int(jtype)})."
            )

        if angvel_world is not None:
            self._data.qvel[vadr : vadr + 3] = np.asarray(
                angvel_world, dtype=float
            ).reshape(3)
        if linvel_world is not None:
            self._data.qvel[vadr + 3 : vadr + 6] = np.asarray(
                linvel_world, dtype=float
            ).reshape(3)

    def set_joint_qpos(self, joint: Union[str, int], value: float):
        jid = self.joint_id(joint)
        adr = self._model.jnt_qposadr[jid]
        # Hinge, slide -> scalar
        self._data.qpos[adr] = float(value)

    def set_joint_qvel(self, joint: Union[str, int], value: float):
        jid = self.joint_id(joint)
        adr = self._model.jnt_dofadr[jid]
        self._data.qvel[adr] = float(value)

    # -------------------- Actuators / controls --------------------
    def set_ctrl(self, actuator: Union[str, int], value: float):
        aid = self.actuator_id(actuator)
        self._data.ctrl[aid] = float(value)

    def set_ctrls(self, values_by_name: Dict[str, float]):
        for name, val in values_by_name.items():
            self.set_ctrl(name, val)

    def actuators_for_body(self, body: int | str):
        """
        Return actuator ids and names that act on the given body.
        Works for JOINT-driven and SITE-driven actuators.
        """
        if isinstance(body, str):
            body_id = self.body_id(body)
        else:
            body_id = int(body)

        mdl = self._model
        ids, names = [], []

        for aid in range(mdl.nu):
            trntype = int(mdl.actuator_trntype[aid])
            idx0 = int(mdl.actuator_trnid[aid, 0])

            # JOINT-driven
            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                if 0 <= idx0 < mdl.njnt and mdl.jnt_bodyid[idx0] == body_id:
                    ids.append(aid)
                    names.append(mdl.actuator(aid).name)
                continue

            # SITE-driven (e.g. quadrotor motors)
            if trntype == mujoco.mjtTrn.mjTRN_SITE:
                if 0 <= idx0 < mdl.nsite and mdl.site_bodyid[idx0] == body_id:
                    ids.append(aid)
                    names.append(mdl.actuator(aid).name)
                continue

        return ids, names

    # -------------------- Convenience: quick “hover” gravity tweak --------------------
    def set_gravity_scale(self, scale: float):
        """Scale current gravity vector magnitude by 'scale' (sign preserved)."""
        g = self._model.opt.gravity.copy()
        mag = np.linalg.norm(g)
        if mag > 0:
            self._model.opt.gravity[:] = g * scale

    # -------------------- Debug helpers --------------------
    def world_linvel_of_body(self, body: Union[str, int]) -> np.ndarray:
        """Return world-frame COM linear velocity of a body (derived from cvel)."""
        bid = self.body_id(body)
        r_bw = self._data.xmat[bid].reshape(3, 3)
        v_lin_body = self._data.cvel[bid, 3:]  # body frame
        return r_bw @ v_lin_body

    def world_pos_of_body(self, body: Union[str, int]) -> np.ndarray:
        return self._data.xpos[self.body_id(body)].copy()


ENVIRONMENT = ENV(CONFIG["path_to_xml"])
