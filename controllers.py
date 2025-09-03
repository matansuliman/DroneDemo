import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

from logger import LOGGER
from config import CONFIG


class BasicController:
    def __init__(self, env, plant):
        self._env = env
        self._plant = plant
        self._log = dict()

    @property
    def log(self):
        return self._log

    def clear_log(self):
        for key in self._log.keys():
            self._log[key] = []

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


def sym_limits(x): return -x, x

class QuadrotorController(BasicController):
    def __init__(self, env, quadrotor):
        super().__init__(env= env, plant= quadrotor)

        self._log = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'x_noise': [], 'y_noise': [], 'z_noise': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []
        }

        conf_pid = CONFIG["QuadrotorController"]["pids"]
        self.pids = {
            'x':     PID(Kp= conf_pid["x"]["ki"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'y':     PID(Kp= conf_pid["y"]["ki"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'z':     PID(Kp= conf_pid["z"]["ki"], Kd= conf_pid["x"]["kd"], setpoint=conf_pid["x"]["setpoint"], output_limits=sym_limits(conf_pid["x"]["output_limit"])),
            'roll':  PID(Kp= conf_pid["ro"]["ki"], Kd= conf_pid["ro"]["kd"], setpoint=conf_pid["ro"]["setpoint"], output_limits=sym_limits(conf_pid["ro"]["output_limit"])),
            'pitch': PID(Kp= conf_pid["pi"]["ki"], Kd= conf_pid["pi"]["kd"], setpoint=conf_pid["pi"]["setpoint"], output_limits=sym_limits(conf_pid["pi"]["output_limit"])),
            'yaw':   PID(Kp= conf_pid["ya"]["ki"], Kd= conf_pid["ya"]["kd"], setpoint=conf_pid["ya"]["setpoint"], output_limits=sym_limits(conf_pid["ya"]["output_limit"])),
        }

        conf_ff = CONFIG["QuadrotorController"]["ff"]
        self.ff = {
            'x': conf_ff["x"],
            'y': conf_ff["y"],
            'z': conf_ff["z"],
            'throttle': conf_ff["throttle"]
        }

        self._descending = False
        LOGGER.info(f"\t\t\tController: Initiated {self.__class__.__name__}")

    @property
    def descending(self):
        return self._descending

    def descend(self):
        self._descending = True
        self.ff['z'] = 0
        self.pids['z'].output_limits = (-0.20, 0.10)
        LOGGER.debug("QuadrotorController: descending")

    def set_reference(self, pos= (0, 0, 0), vel= None):
        if vel is None:
            self._env.set_free_body_pose(self._plant.XML_BODY_NAME, pos_world=pos, quat_wxyz=[1, 0, 0, 0])
            self._env.set_free_body_velocity(self._plant.XML_BODY_NAME, linvel_world=[0, 0, 0], angvel_world=[0, 0, 0])
        else:
            # Update PID setpoints
            self.pids['x'].setpoint = pos[0] + self.ff['x'] * vel[0]
            self.pids['y'].setpoint = pos[1] + self.ff['y'] * vel[1]
            self.pids['z'].setpoint = pos[2] + self.ff['z']

    def _outer_loop(self):
        pos = self._plant.get_pos(mode='no_noise')
        noise_pos = self._plant.get_pos(mode='noise')

        # Log
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])

        self._log['x_noise'].append(noise_pos[0])
        self._log['y_noise'].append(noise_pos[1])
        self._log['z_noise'].append(noise_pos[2])

        hz = 20
        interval = self._env.dt * hz
        current_time = self._env.get_time()
        previous_time = current_time - self._env.dt

        if (current_time // interval) != (previous_time // interval):
            self.pids['pitch'].setpoint = self.pids['x'](pos[0])
            self.pids['roll'].setpoint = -self.pids['y'](pos[1])

        # Altitude throttle
        throttle = self.pids['z'](pos[2]) + self.ff['throttle']
        return throttle

    def _inner_loop(self):
        # Convert quaternion to Euler angles
        quat_wxyz = self._env.data.xquat[self._plant.body_id]
        roll, pitch, yaw = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler('xyz', degrees=False)

        # Log
        self._log['roll'].append(roll)
        self._log['pitch'].append(pitch)
        self._log['yaw'].append(yaw)

        # Control signals
        roll_cmd = self.pids['roll'](roll)
        pitch_cmd = self.pids['pitch'](pitch)
        yaw_cmd = self.pids['yaw'](yaw)

        # Log
        self._log['roll_cmd'].append(roll_cmd)
        self._log['pitch_cmd'].append(pitch_cmd)
        self._log['yaw_cmd'].append(yaw_cmd)

        return roll_cmd, pitch_cmd, yaw_cmd

    def _apply_cmds(self, throttle, roll_cmd, pitch_cmd, yaw_cmd):
        # Motor Mixing Algorithm (MMA): Combine control signals to actuators
        values = {
            'thrust1': throttle + roll_cmd + pitch_cmd + yaw_cmd,
            'thrust2': throttle - roll_cmd + pitch_cmd - yaw_cmd,
            'thrust3': throttle - roll_cmd - pitch_cmd + yaw_cmd,
            'thrust4': throttle + roll_cmd - pitch_cmd - yaw_cmd,
        }
        self._env.set_ctrls(values)

    def step(self):

        # Log
        self._log['time'].append(self._env.get_time())

        # Outer loop position control
        throttle = self._outer_loop()

        # Inner loop orientation control
        roll_cmd, pitch_cmd, yaw_cmd = self._inner_loop()

        # Apply control signals to actuators
        self._apply_cmds(throttle, roll_cmd, pitch_cmd, yaw_cmd)


DEFAULT_VELOCITY = (0, 0, 0)

class MovingPlatformController(BasicController):
    def __init__(self, env, platform):
        super().__init__(env= env, plant= platform)

        self._velocity = np.array(DEFAULT_VELOCITY, dtype=np.float64)
        self._locks_activated = False

        self._log = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'x_noise': [], 'y_noise': [], 'z_noise': [],
        }
        LOGGER.info(f"\t\t\tController: Initiated {self.__class__.__name__}")

    @property
    def locks_activated(self):
        return self._locks_activated

    def activate_locks(self):
        self._locks_activated = True
        LOGGER.debug("MovingPlatformController: activating locks")

    def deactivate_locks(self):
        self._locks_activated = False

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = np.array(velocity, dtype=np.float64)

    def step(self):
        # Update position based on velocity
        pos = self._plant.get_pos(mode='no_noise')
        new_pos = pos + self._velocity * self._env.dt
        self._env.set_joint_qpos(self._plant.joint_x_name, float(new_pos[0]))
        self._env.set_joint_qpos(self._plant.joint_y_name, float(new_pos[1]))

        # Logging
        self._log['time'].append(self._env.get_time())
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])
        pos_noise = self._plant.get_pos(mode='noise')
        self._log['x_noise'].append(pos_noise[0])
        self._log['y_noise'].append(pos_noise[1])
        self._log['z_noise'].append(pos_noise[2])