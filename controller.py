# PATCH in controller.py

import mujoco
import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

class QuadrotorController:
    def __init__(self, model_path: str, 
                 target: np.ndarray, 
                 velocity: np.ndarray, 
                 dt: float = 0.01):

        def _init_pid_controllers():
            # Outer loop: Position controllers for x, y, and z.
            self.pid_x = PID(0.15, 0.00, 0.40, setpoint=0, output_limits=(-0.03, 0.03))
            self.pid_y = PID(0.15, 0.00, 0.40, setpoint=0, output_limits=(-0.03, 0.03))
            self.pid_z = PID(0.15, 0.00, 0.80, setpoint=1, output_limits=(-0.10, 0.10))
            self.gravity_ff = 3.2495625

            # Inner loop: Attitude controllers for roll, pitch, and yaw.
            self.pid_roll  = PID(0.50, 0.00, 0.30, setpoint=0, output_limits=(-0.50, 0.50))
            self.pid_pitch = PID(0.50, 0.00, 0.20, setpoint=0, output_limits=(-0.50, 0.50))
            self.pid_yaw   = PID(0.50, 0.00, 0.50, setpoint=0, output_limits=(-2.00, 2.00))

        def _map_motors():
            # Map motor actuator names to indices
            motor_names = [self.model.actuator(i).name for i in range(self.model.nu)]
            return {name: i for i, name in enumerate(motor_names)}

        self.dt = dt
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.target = np.array(target, dtype=np.float64)
        
        self.platform_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

        _init_pid_controllers()
        self.motor_map = _map_motors()
        self.body_id = next(i for i in range(self.model.nbody) if self.model.body(i).name == 'x2')
        self.platform_x_id = self.model.joint("platform_x").qposadr
        self.platform_y_id = self.model.joint("platform_y").qposadr

        self.target_z_ff = 0.5


        self.log = {'time': [],
                    'x': [], 'y': [], 'z': [],
                    'roll': [], 'pitch': [], 'yaw': [],
                    'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []}

        self.paused = False
        self.terminated = False

    def update_target(self):
        # Move the platform
        self.platform_pos += self.velocity[:3] * self.dt
        self.data.qpos[self.platform_x_id] = self.platform_pos[0]
        self.data.qpos[self.platform_y_id] = self.platform_pos[1]

        # Drone target follows the platform + 0.5m in Z
        self.target = np.array([
            self.platform_pos[0],
            self.platform_pos[1],
            self.platform_pos[2]
        ])

        # Feedforward terms
        target_x_ff = 1.5 * self.velocity[0]
        target_y_ff = 1.5 * self.velocity[1]

        self.pid_x.setpoint = self.target[0] + target_x_ff
        self.pid_y.setpoint = self.target[1] + target_y_ff
        self.pid_z.setpoint = self.target[2] + self.target_z_ff

    def step(self):

        def _outer_loop():
            pos = self.data.xpos[self.body_id]

            #Log
            self.log['x'].append(pos[0])
            self.log['y'].append(pos[1])
            self.log['z'].append(pos[2])

            if self.data.time // (self.dt * 10) != (self.data.time - self.dt) // (self.dt * 10):
                self.pid_pitch.setpoint = self.pid_x(pos[0])
                self.pid_roll.setpoint = -self.pid_y(pos[1])

            throttle = self.pid_z(pos[2]) + self.gravity_ff
            return throttle

        def _inner_loop():
            quat = self.data.qpos[3:7]
            roll, pitch, yaw = R.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz', degrees=False) # Convert quaternion to Euler angles

            # Log
            self.log['roll'].append(roll)
            self.log['pitch'].append(pitch)
            self.log['yaw'].append(yaw)

            # Control signals
            roll_cmd = self.pid_roll(roll)
            pitch_cmd = self.pid_pitch(pitch)
            yaw_cmd = self.pid_yaw(yaw)

            # Log
            self.log['roll_cmd'].append(roll_cmd)
            self.log['pitch_cmd'].append(pitch_cmd)
            self.log['yaw_cmd'].append(yaw_cmd)

            return roll_cmd, pitch_cmd, yaw_cmd

        def _apply_cmds():
            d = self.data.ctrl
            m = self.motor_map
            # Combine control signals to actuators
            d[m['thrust1']] = throttle + roll_cmd + pitch_cmd + yaw_cmd
            d[m['thrust2']] = throttle - roll_cmd + pitch_cmd - yaw_cmd
            d[m['thrust3']] = throttle - roll_cmd - pitch_cmd + yaw_cmd
            d[m['thrust4']] = throttle + roll_cmd - pitch_cmd - yaw_cmd

        mujoco.mj_step(self.model, self.data)

        # Log
        self.log['time'].append(self.data.time)

        # Outer loop position control
        throttle = _outer_loop()

        # Inner loop orientation control
        roll_cmd, pitch_cmd, yaw_cmd = _inner_loop()

        # Apply control signals to actuators
        _apply_cmds()