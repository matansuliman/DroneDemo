import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation as R


class BasicController:
    def __init__(self, info: str ='BasicController', env= None, plant= None):
        self._info = info
        self._env = env
        self._plant = plant
        self._log = {}

    @property
    def info(self):
        return self._info

    @property
    def log(self):
        return self._log
    
    def clear_log(self):
        for key in self._log.keys():
            self._log[key] = [] 
    
    def step(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self):
        return f'model ({self.__class__.__name__}) info: {self._info}'


def lower_upper_limits(x): return -x, x

CMD_ANGLE_LIMITS = lower_upper_limits(0.03)
ANGLE_LIMITS = lower_upper_limits(0.05)
ALT_LIMITS = lower_upper_limits(0.8)
ALT_FF = 3.2495625

HOVER_ALT_FF = 10

OUTERLOOP_RATE = 20
VELOCITY_FF_COEF = 1.74

LANDING_ALT_LIMITS = (-0.20, 0.10)
LANDING_ALT_FF = 0


class QuadrotorController(BasicController):
    def __init__(self, info: str='QuadrotorController', env= None, quadrotor= None):
        super().__init__(
            info=info,
            env=env,
            plant=quadrotor,
        )

        self._log = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'x_noise': [], 'y_noise': [], 'z_noise': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []
        }
        
        self.pids = {
            'x': PID(Kp=0.15, Kd=0.40, setpoint=0, output_limits=CMD_ANGLE_LIMITS),
            'y': PID(Kp=0.15, Kd=0.40, setpoint=0, output_limits=CMD_ANGLE_LIMITS),
            'z': PID(Kp=0.15, Kd=0.55, setpoint=1, output_limits=ALT_LIMITS),
            'roll': PID(Kp=0.50, Kd=0.30, setpoint=0, output_limits=ANGLE_LIMITS),
            'pitch': PID(Kp=0.50, Kd=0.20, setpoint=0, output_limits=ANGLE_LIMITS),
            'yaw': PID(Kp=0.50, Kd=0.50, setpoint=0, output_limits=(-2.00, 2.00))
        }

        self.ff = {
            'target_x': 0,
            'target_y': 0,
            'target_z': HOVER_ALT_FF,
            'gravity': ALT_FF
        }

        # Initialize target
        self._target = np.zeros(3, dtype=np.float64)

        self._descending = False
    
    @property
    def target(self):
        return self._target

    @property
    def descending(self):
        return self._descending

    def descend(self):
        self._descending = True
        self.ff['target_z'] = LANDING_ALT_FF
        self.pids['z'].output_limits = LANDING_ALT_LIMITS

    def update_target(self, mode, data=None):
        if mode == 'hardcode':
            self._env.set_free_body_pose('x2', pos_world=data['qpos'], quat_wxyz=[1, 0, 0, 0])
            self._env.set_free_body_velocity('x2', linvel_world=[0, 0, 0], angvel_world=[0, 0, 0])
        
        elif mode == 'follow': # Drone target follows the platform + 0.5m in Z
            self._target = data['new_target_pos'].copy() 

            # Feedforward terms
            self.ff['target_x'] = VELOCITY_FF_COEF * data['new_target_vel'][0]
            self.ff['target_y'] = VELOCITY_FF_COEF * data['new_target_vel'][1]

            # Update PID setpoints
            self.pids['x'].setpoint = self._target[0] + self.ff['target_x']
            self.pids['y'].setpoint = self._target[1] + self.ff['target_y']
            self.pids['z'].setpoint = self._target[2] + self.ff['target_z']
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _outer_loop(self, fused_pos=None):

        pos = self._plant.get_pos(mode='no_noise')
        noise_pos = self._plant.get_pos(mode='noise')
        noise_pos = pos

        # Log
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])

        
        self._log['x_noise'].append(noise_pos[0])
        self._log['y_noise'].append(noise_pos[1])
        self._log['z_noise'].append(noise_pos[2])

        interval = self._env.dt * OUTERLOOP_RATE
        current_time = self._env.get_time()
        previous_time = current_time - self._env.dt
        
        if  (current_time // interval) != (previous_time // interval):
            self.pids['pitch'].setpoint = self.pids['x'](noise_pos[0])
            self.pids['roll'].setpoint = -self.pids['y'](noise_pos[1])

        # Altitude throttle
        throttle = self.pids['z'](noise_pos[2]) + self.ff['gravity']
        
        return throttle
    
    def _inner_loop(self):
        # [w, x, y, z]
        quat = self._env.data.xquat[self._plant.body_id]
        
        # Convert quaternion to Euler angles
        roll, pitch, yaw = R.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz', degrees=False)
        
        # Log
        self.log['roll'].append(roll)
        self.log['pitch'].append(pitch)
        self.log['yaw'].append(yaw)
        
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
        # Motor Mixing Algorithem (MMA): Combine control signals to actuators
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



DEFAULT_VELOCITY = (0.0, 0.0, 0.0)

class MovingPlatformController(BasicController):
    def __init__(self, info: str='QuadrotorController', env= None, platform= None):
        super().__init__(
            info=info,
            env=env,
            plant=platform,
        )

        self._velocity = np.array(DEFAULT_VELOCITY, dtype=np.float64)
        self._locks_activated = False

        self._log = {
            'time': [], 
            'x': [], 'y': [], 'z': [],
            'x_noise': [], 'y_noise': [], 'z_noise': [],
        }
    
    @property
    def locks_activated(self):
        return self._locks_activated

    def activate_locks(self):
        self._locks_activated = True

    def deactivate_locks(self):
        self._locks_activated = False

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

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
