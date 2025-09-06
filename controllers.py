import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation

from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicController:
    def __init__(self, plant):
        self._plant = plant
        self._log = dict()

    @property
    def log(self):
        return self._log

    def clear_log(self):
        for key in self._log.keys():
            self._log[key] = []

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class QuadrotorController(BasicController):

    def __init__(self, quadrotor):
        super().__init__(plant= quadrotor)

        self._log = {
            'time': [],
            'x_true': [], 'y_true': [], 'z_true': [],
            'x': [], 'y': [], 'z': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []
        }

        conf_pid = CONFIG["QuadrotorController"]["pids"]
        self._pids = {
            'x':     PID(Kp= conf_pid["x"]["kp"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'y':     PID(Kp= conf_pid["y"]["kp"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'z':     PID(Kp= conf_pid["z"]["kp"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'roll':  PID(Kp= conf_pid["ro"]["kp"], Kd= conf_pid["ro"]["kd"], setpoint=conf_pid["ro"]["setpoint"], output_limits=sym_limits(conf_pid["ro"]["output_limit"])),
            'pitch': PID(Kp= conf_pid["pi"]["kp"], Kd= conf_pid["pi"]["kd"], setpoint=conf_pid["pi"]["setpoint"], output_limits=sym_limits(conf_pid["pi"]["output_limit"])),
            'yaw':   PID(Kp= conf_pid["ya"]["kp"], Kd= conf_pid["ya"]["kd"], setpoint=conf_pid["ya"]["setpoint"], output_limits=sym_limits(conf_pid["ya"]["output_limit"])),
        }

        conf_ff = CONFIG["QuadrotorController"]["ff"]
        self._ff = {
            'x': conf_ff["x"],
            'y': conf_ff["y"],
            'z': conf_ff["z"],
            'throttle': conf_ff["throttle"]
        }

        self._reference = None

        self._descending = False
        descend_phases = CONFIG["QuadrotorController"]["descend_phases"]
        self._descend_phases = list(zip(descend_phases["names"], descend_phases["ff_z"], descend_phases["pid_z_ul"], descend_phases["pid_z_ll"]))

        LOGGER.info(f"\t\t\tController: Initiated {self.__class__.__name__}")

    @property
    def descending(self):
        return self._descending

    def descend(self):
        self._descending = True
        LOGGER.debug("QuadrotorController: descending")

    def teleport(self, pos):
        ENVIRONMENT.set_free_body_pose(self._plant.xml_name, pos_world= pos, quat_wxyz= [1, 0, 0, 0])
        ENVIRONMENT.set_free_body_velocity(self._plant.xml_name, linvel_world= [0, 0, 0], angvel_world= [0, 0, 0])

    def set_reference(self, pos, vel):
        self._reference = (pos, vel)
        # Update PID setpoints
        self._pids['x'].setpoint = pos[0] + self._ff['x'] * vel[0]
        self._pids['y'].setpoint = pos[1] + self._ff['y'] * vel[1]
        self._pids['z'].setpoint = pos[2] + self._ff['z']

    def get_reference(self):
        return self._reference

    def _outer_loop(self):
        pos_true = self._plant.get_true_pos()
        pos = self._plant.get_pos()

        # Log
        self._log['x_true'].append(pos_true[0])
        self._log['y_true'].append(pos_true[1])
        self._log['z_true'].append(pos_true[2])

        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])

        hz = 20
        interval = ENVIRONMENT.dt * hz
        current_time = ENVIRONMENT.get_time()
        previous_time = current_time - ENVIRONMENT.dt

        if (current_time // interval) != (previous_time // interval):
            self._pids['pitch'].setpoint = self._pids['x'](pos_true[0])
            self._pids['roll'].setpoint = -self._pids['y'](pos_true[1])

        # Altitude throttle
        throttle = self._pids['z'](pos_true[2]) + self._ff['throttle']
        return throttle

    def _inner_loop(self):
        # Convert quaternion to Euler angles
        quat_wxyz = ENVIRONMENT.data.xquat[self._plant.body_id]
        roll, pitch, yaw = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler('xyz', degrees=False)

        # Log
        self._log['roll'].append(roll)
        self._log['pitch'].append(pitch)
        self._log['yaw'].append(yaw)

        # Control signals
        roll_cmd = self._pids['roll'](roll)
        pitch_cmd = self._pids['pitch'](pitch)
        yaw_cmd = self._pids['yaw'](yaw)

        # Log
        self._log['roll_cmd'].append(roll_cmd)
        self._log['pitch_cmd'].append(pitch_cmd)
        self._log['yaw_cmd'].append(yaw_cmd)

        return roll_cmd, pitch_cmd, yaw_cmd

    def _apply_cmds(self, throttle, roll_cmd, pitch_cmd, yaw_cmd):
        # Motor Mixing Algorithm (MMA): Combine control signals to actuators
        M = np.array(CONFIG["QuadrotorController"]["mixing-matrix"]) # mixing matrix
        u = np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd]) # control vector
        thrusts = M @ u  # compute thrusts

        values = {
            'thrust1': thrusts[0],
            'thrust2': thrusts[1],
            'thrust3': thrusts[2],
            'thrust4': thrusts[3],
        }

        #print(values)

        ENVIRONMENT.set_ctrls(values)

    def _get_phase(self):
        epsilon = 0.05
        val = self._plant.sensors['rangefinder'].get()
        for name, ff_z, pid_z_ul, pid_z_ll in self._descend_phases:
            if val > ff_z + epsilon:
                return name, ff_z, pid_z_ul, pid_z_ll

        return 'clear_all', 0, 0, 0

    def _enforce_descend(self):
        _, ff_z, pid_z_ul, pid_z_ll = self._get_phase()
        self._ff['z'] = ff_z
        self._pids['z'].output_limits = pid_z_ll, pid_z_ul

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"\treference_pos: {print_array_of_nums(self.get_reference()[0])}\n"
        status += f"\treference_vel: {print_array_of_nums(self.get_reference()[1])}\n"
        if self._descending: status += f"\tdescend_phase: {self._get_phase()[0]}\n"
        return status

    def step(self):
        # Enforce descend
        if self._descending: self._enforce_descend()

        if self._get_phase()[0] in ['clear_all']:
            ENVIRONMENT.data.ctrl[:] = 0
            ENVIRONMENT.data.act[:] = 0
            return

        # Outer loop position control -> throttle
        # Inner loop orientation control -> (roll_cmd, pitch_cmd, yaw_cmd)
        # Apply control signals to actuators
        self._log['time'].append(ENVIRONMENT.get_time())  # Log
        self._apply_cmds(self._outer_loop(), *self._inner_loop())


class PadController(BasicController):
    def __init__(self, pad):
        super().__init__(plant= pad)

        self._velocity = np.array(CONFIG["Pad"]["default_velocity"], dtype=np.float64)
        self._locks_activated = False

        self._log = {
            'time': [],
            'x_true': [], 'y_true': [], 'z_true': [],
            'x': [], 'y': [], 'z': [],
        }
        LOGGER.info(f"\t\t\tPadController: Initiated {self.__class__.__name__}")

    @property
    def locks_activated(self):
        return self._locks_activated

    def activate_locks(self, pos):
        self._locks_activated = True
        self._plant.locks_end_pos = pos
        LOGGER.debug("PadController: activating locks")

    def deactivate_locks(self):
        self._locks_activated = False

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = np.array(velocity, dtype=np.float64)

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"\tvel: {self.velocity}\n"
        status += f"\tlocks_activated: {self._locks_activated}\n"
        return status

    def step(self):
        # Update position based on velocity
        pos_true = self._plant.get_true_pos()
        new_pos = pos_true + self._velocity * ENVIRONMENT.dt
        ENVIRONMENT.set_joint_qpos(self._plant.joint_x_name, float(new_pos[0]))
        ENVIRONMENT.set_joint_qpos(self._plant.joint_y_name, float(new_pos[1]))

        # Logging
        self._log['time'].append(ENVIRONMENT.get_time())
        self._log['x_true'].append(pos_true[0])
        self._log['y_true'].append(pos_true[1])
        self._log['z_true'].append(pos_true[2])
        pos = self._plant.get_pos()
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])