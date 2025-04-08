import numpy as np
from simple_pid import PID
from scipy.spatial.transform import Rotation as R

from sensors import GPS, INS

XML_NAME = 'x2'

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

class QuadrotorController:
    def __init__(self,
                 model, 
                 data,
                 dt: float = 0.01):
        
        def _init_pid_controllers():
            # Outer loop: Position controllers for x, y, and z.
            self.pid_x = PID(Kp=0.15, Kd=0.42, setpoint=0, output_limits=CMD_ANGLE_LINITS)
            self.pid_y = PID(Kp=0.15, Kd=0.42, setpoint=0, output_limits=CMD_ANGLE_LINITS)
            self.pid_z = PID(Kp=0.15, Kd=0.55, setpoint=1, output_limits=ALT_LIMITS)
            self.gravity_ff = ALT_FF

            # Inner loop: Attitude controllers for roll, pitch, and yaw.
            self.pid_roll  = PID(Kp=0.50, Kd=0.30, setpoint=0, output_limits=ANGLE_LINITS)
            self.pid_pitch = PID(Kp=0.50, Kd=0.20, setpoint=0, output_limits=ANGLE_LINITS)
            self.pid_yaw   = PID(Kp=0.50, Kd=0.50, setpoint=0, output_limits=(-2.00, 2.00))
            
            """ for check yaw control
            self.pid_roll  = PID(Kp=0, Kd=0, setpoint=0, output_limits=ANGLE_LINITS)
            self.pid_pitch = PID(Kp=0, Kd=0, setpoint=0, output_limits=ANGLE_LINITS)
            """

        def _map_motors():
            # Map motor actuator names to indices
            motor_names = [self.model.actuator(i).name for i in range(self.model.nu)]
            return {name: i for i, name in enumerate(motor_names)}

        self.model = model
        self.data = data
        self.dt = dt

        # init body id
        self.body_id = next(i for i in range(self.model.nbody) if self.model.body(i).name == XML_NAME)

        # Initialize sensors
        self.gps = GPS(data, self.body_id, dt)
        self.ins = INS(data, self.body_id, dt) # internal navigation system

        self.target = np.zeros(3, dtype=np.float64)
        self.target_z_ff = HOVER_ALT_FF

        _init_pid_controllers()
        self.motor_map = _map_motors()
        

        self.accel_sensor_id = next(i for i in range(self.model.nsensor) if self.model.sensor(i).name == "body_linacc")
        self.gyro_sensor_id = next(i for i in range(self.model.nsensor) if self.model.sensor(i).name == "body_gyro")
        
        self.log = {'time': [],
                    'x': [], 'y': [], 'z': [],
                    'roll': [], 'pitch': [], 'yaw': [],
                    'roll_cmd': [], 'pitch_cmd': [], 'yaw_cmd': []}
        
        self.terminated = False
        self.paused = False 
        self.lock = False

    def apply_landing(self):
        self.target_z_ff = LANDING_ALT_FF
        self.pid_z.output_limits = LANDING_ALT_LIMITS

    def update_target(self, new_target_pos: np.ndarray, new_target_vel: np.ndarray):
        def _print_error():
            pos = self.data.xpos[self.body_id]
            error_x = pos[0] - self.target[0]
            error_y = pos[1] - self.target[1]
            error_z = pos[2] - self.target[2]

            print(f"error_x: {error_x:.2f} error_y: {error_y:.2f} error_z: {error_z:.2f}", end='\r')

        # Drone target follows the platform + 0.5m in Z
        self.target = new_target_pos.copy()

        #_print_error()

        # Feedforward terms
        target_x_ff = VELOCITY_FF_COEF * new_target_vel[0]
        target_y_ff = VELOCITY_FF_COEF * new_target_vel[1]

        """
        pos = self.gps.get_true_pos()
        error_x = pos[0] - self.target[0]
        error_y = pos[1] - self.target[1]
        if new_target_vel[0] != 0 and new_target_vel[1] != 0:
            print(f"{error_x:.3f}, {error_x/new_target_vel[0]:.2f} \t{error_y:.3f}, {error_y/new_target_vel[1]:.2f}", end='\r')
        """

        # Update PID setpoints
        self.pid_x.setpoint = self.target[0] + target_x_ff
        self.pid_y.setpoint = self.target[1] + target_y_ff
        self.pid_z.setpoint = self.target[2] + self.target_z_ff

        """ for yaw control
        delta = new_target_pos[:2] - self.get_true_pos()[:2]
        print(np.arctan2(delta[1], delta[0]), end='\r')
        self.pid_yaw.setpoint = np.arctan2(delta[1], delta[0])
        """

    def step(self):

        def _outer_loop():

            def _pos_rel_body():
                # Compute position error in world frame
                pos_error_world = self.target[:2] - pos[:2]

                # Rotate world error into body frame using inverse yaw
                c, s = np.cos(-yaw), np.sin(-yaw)
                R_world_to_body = np.array([[c, -s], [s, c]])
                pos_error_body = R_world_to_body @ pos_error_world
                #print(f"pos_error_body: {pos_error_body[0]:.2f} {pos_error_body[1]:.2f}", end='\r')
                return pos_error_body

            pos = self.gps.get_true_pos()
            quat = self.data.qpos[3:7]
            yaw = R.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')[2]

            # Log
            self.log['x'].append(pos[0])
            self.log['y'].append(pos[1])
            self.log['z'].append(pos[2])
            
            if self.data.time // (self.dt * OUTERLOOP_RATE) != (self.data.time - self.dt) // (self.dt * OUTERLOOP_RATE):
                """ for yaw control
                pos_rel_body = _pos_rel_body()

                # Set setpoints in body frame direction
                self.pid_pitch.setpoint = self.pid_x(pos_rel_body[0])
                self.pid_roll.setpoint = -self.pid_y(pos_rel_body[1])
                """
                self.pid_pitch.setpoint = self.pid_x(pos[0])
                self.pid_roll.setpoint = -self.pid_y(pos[1])

            # Altitude throttle
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
            # Motor Mixing Algorithem (MMA): Combine control signals to actuators
            d[m['thrust1']] = throttle + roll_cmd + pitch_cmd + yaw_cmd
            d[m['thrust2']] = throttle - roll_cmd + pitch_cmd - yaw_cmd
            d[m['thrust3']] = throttle - roll_cmd - pitch_cmd + yaw_cmd
            d[m['thrust4']] = throttle + roll_cmd - pitch_cmd - yaw_cmd

        # Log
        self.log['time'].append(self.data.time)

        """
        def _ins_update():
            accel_data = self.data.sensordata[self.accel_sensor_id : self.accel_sensor_id + 3]  # Accelerometer data (x, y, z)
            gyro_data = self.data.sensordata[self.gyro_sensor_id : self.gyro_sensor_id + 3]  # Gyroscope data (roll rate, pitch rate, yaw rate)
            #with open('data.txt', 'a') as f:
            #    f.write(f"{accel_data[0]:.5f} {accel_data[1]:.5f} {accel_data[2]:.5f} {gyro_data[0]:.5f} {gyro_data[1]:.5f} {gyro_data[2]:.5f}\n")
            #print(f"accel_data: {accel_data[0]:.5f} {accel_data[1]:.5f} {accel_data[2]:.5f}, gyro_data: {gyro_data[0]:.5f} {gyro_data[1]:.5f} {gyro_data[2]:.5f}", end='\r')
            self.ins.update(accel_data, gyro_data) # Update INS (integrate the data)

            position = self.gps.get_true_pos()
            print(
                f"INS: {self.ins} ",
                f"True position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}" , 
                f"True velocity: {self.data.qvel[0]:.4f}, {self.data.qvel[1]:.4f}, {self.data.qvel[2]:.4f}" , 
                end='\r')

        _ins_update()
        """

        # Outer loop position control
        throttle = _outer_loop()

        # Inner loop orientation control
        roll_cmd, pitch_cmd, yaw_cmd = _inner_loop()

        # Apply control signals to actuators
        _apply_cmds()