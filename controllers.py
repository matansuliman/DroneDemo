import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

CMD_ANGLE_LINIT = 0.03
CMD_ANGLE_LINITS = (-CMD_ANGLE_LINIT, CMD_ANGLE_LINIT)

ANGLE_LINIT = 0.05
ANGLE_LINITS = (-ANGLE_LINIT, ANGLE_LINIT)

ALT_LIMIT = 0.8
ALT_LIMITS = (-ALT_LIMIT, ALT_LIMIT)
ALT_FF = 3.2495625

HOVER_ALT_FF = 10

OUTERLOOP_RATE = 20
VELOCITY_FF_COEF = 1.74

LANDING_ALT_LIMITS = (-0.20, 0.10)
LANDING_ALT_FF = 0.02

class basicController:
    def __init__(self, env, plant):
        self._env = env
        self._plant = plant
        self._log = None

    @property
    def log(self):
        return self._log
    
    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class QuadrotorController(basicController):
    def __init__(self, env, drone):
        super().__init__(env, drone)

        self._log = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []
        }
        
        self.pids = {
            'x': PID(Kp=0.15, Kd=0.42, setpoint=0, output_limits=CMD_ANGLE_LINITS),
            'y': PID(Kp=0.15, Kd=0.42, setpoint=0, output_limits=CMD_ANGLE_LINITS),
            'z': PID(Kp=0.15, Kd=0.55, setpoint=1, output_limits=ALT_LIMITS),
            'roll': PID(Kp=0.50, Kd=0.30, setpoint=0, output_limits=ANGLE_LINITS),
            'pitch': PID(Kp=0.50, Kd=0.20, setpoint=0, output_limits=ANGLE_LINITS),
            'yaw': PID(Kp=0.50, Kd=0.50, setpoint=0, output_limits=(-2.00, 2.00))
        }

        self.ff = {
            'target_x': 0,
            'target_y': 0,
            'target_z': HOVER_ALT_FF,
            'gravity': ALT_FF
        }

        # Initialize target position
        self._target = np.zeros(3, dtype=np.float64)
    
    @property
    def target(self):
        return self._target

    def update_target(self, mode, data=None):
        if mode == 'hardcode':
            self._env.data.qpos[:3] = data['qpos']
            self._env.data.qvel[:] = 0
            self._env.data.qacc[:] = 0
            self._env.data.qpos[3:7] = [1, 0, 0, 0]  # upright orientation
        
        elif mode == 'follow': # Drone target follows the platform + 0.5m in Z
            self._target = data['new_target_pos'].copy() 

            # Feedforward terms
            self.ff['target_x'] = VELOCITY_FF_COEF * data['new_target_vel'][0]
            self.ff['target_y'] = VELOCITY_FF_COEF * data['new_target_vel'][1]

            # Update PID setpoints
            self.pids['x'].setpoint = self._target[0] + self.ff['target_x']
            self.pids['y'].setpoint = self._target[1] + self.ff['target_y']
            self.pids['z'].setpoint = self._target[2] + self.ff['target_z']
        
        elif mode == 'landing':
            self.ff['target_z'] = LANDING_ALT_FF
            self.pids['z'].output_limits = LANDING_ALT_LIMITS
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _outer_loop(self):

        pos = self._plant.getPos(mode='no_noise')
        noise_pos = self._plant.getPos(mode='noise')
        noise_pos = pos

        # Log
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])

        interval = self._env.dt * OUTERLOOP_RATE
        current_time = self._env.getTime()
        previous_time = current_time - self._env.dt
        
        if  (current_time // interval) != (previous_time // interval):
            self.pids['pitch'].setpoint = self.pids['x'](noise_pos[0])
            self.pids['roll'].setpoint = -self.pids['y'](noise_pos[1])

        # Altitude throttle
        throttle = self.pids['z'](noise_pos[2]) + self.ff['gravity']
        
        return throttle
    
    def _inner_loop(self):
        
        quat = self._env.data.qpos[3:7]
        
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

        d = self._env.data.ctrl
        m = self._plant.motorMap

        # Motor Mixing Algorithem (MMA): Combine control signals to actuators
        d[m['thrust1']] = throttle + roll_cmd + pitch_cmd + yaw_cmd
        d[m['thrust2']] = throttle - roll_cmd + pitch_cmd - yaw_cmd
        d[m['thrust3']] = throttle - roll_cmd - pitch_cmd + yaw_cmd
        d[m['thrust4']] = throttle + roll_cmd - pitch_cmd - yaw_cmd

    def step(self):

        # Log
        self._log['time'].append(self._env.getTime())

        # Outer loop position control
        throttle = self._outer_loop()

        # Inner loop orientation control
        roll_cmd, pitch_cmd, yaw_cmd = self._inner_loop()

        # Apply control signals to actuators
        self._apply_cmds(throttle, roll_cmd, pitch_cmd, yaw_cmd)


"""
def _ins_update():
    accel_data = self.env.data.sensordata[self.accel_sensor_id : self.accel_sensor_id + 3]  # Accelerometer data (x, y, z)
    gyro_data = self.env.data.sensordata[self.gyro_sensor_id : self.gyro_sensor_id + 3]  # Gyroscope data (roll rate, pitch rate, yaw rate)
    #with open('data.txt', 'a') as f:
    #    f.write(f"{accel_data[0]:.5f} {accel_data[1]:.5f} {accel_data[2]:.5f} {gyro_data[0]:.5f} {gyro_data[1]:.5f} {gyro_data[2]:.5f}\n")
    #print(f"accel_data: {accel_data[0]:.5f} {accel_data[1]:.5f} {accel_data[2]:.5f}, gyro_data: {gyro_data[0]:.5f} {gyro_data[1]:.5f} {gyro_data[2]:.5f}", end='\r')
    self.ins.update(accel_data, gyro_data) # Update INS (integrate the data)

    position = self.gps.get_true_pos()
    print(
        f"INS: {self.ins} ",
        f"True position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}" , 
        f"True velocity: {self.env.data.qvel[0]:.4f}, {self.env.data.qvel[1]:.4f}, {self.env.data.qvel[2]:.4f}" , 
        end='\r')

_ins_update()
"""


class MovingPlatformController(basicController):
    def __init__(self, env, platform):
        super().__init__(env, platform)
        self._locks_activated = False

        self._log = {
            'time': [], 
            'x': [], 'y': [], 'z': []
        }
    
        
    @property
    def locks_activated(self):
        return self._locks_activated

    
    def activate_locks(self):
        self._locks_activated = True

    def deactivate_locks(self):
        self._locks_activated = False

    def step(self):
        # Update position based on velocity
        curr_pos = self._plant.getPos(mode='no_noise')
        new_pos = curr_pos + self._plant.velocity * self._env.dt
        self._env.data.qpos[self._plant._joint_x_id] = new_pos[0]
        self._env.data.qpos[self._plant._joint_y_id] = new_pos[1]

        # Logging true position
        self._log['time'].append(self._env.data.time)
        pos = self._plant.getPos(mode='no_noise')
        self._log['x'].append(pos[0])
        self._log['y'].append(pos[1])
        self._log['z'].append(pos[2])
