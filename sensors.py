import numpy as np
from scipy.spatial.transform import Rotation as R
from noise import GPSNoise


class GPS:
    def __init__(self, 
                 data, 
                 body_id: int, 
                 dt: float):
        self.data = data
        self.body_id = body_id
        self.gps_noise = GPSNoise()
        self.dt = dt

    def get_pos(self):
        return self.gps_noise.step(self.data.xpos[self.body_id], self.dt)       
        
    def get_true_pos(self):
        return self.data.xpos[self.body_id]

class INS:
    def __init__(self, 
                 data, 
                 body_id: int, 
                 dt: float):
        
        self.data = data
        self.body_id = body_id
        # Initialize position, velocity, and orientation
        self.position = self.data.xpos[self.body_id]  # Initial position
        self.velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
        self.orientation = np.array([0.0, 0.0, 0.0])  # Euler angles (roll, pitch, yaw)
        self.dt = dt  # Time step

    def update(self, accel_data, gyro_data):
        self.orientation += gyro_data * self.dt # Update orientation

        r = R.from_euler('xyz', self.orientation) # Convert to rotation matrix
        accel_world = r.apply(accel_data) # Rotate acceleration to world frame

        # Integrate
        self.velocity += accel_world * self.dt
        self.position += self.velocity * self.dt

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_orientation(self):
        return self.orientation
    
    def __str__(self):
        return f"Position: {self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f} " \
                f"Vellocity: {self.velocity[0]:.7f}, {self.velocity[1]:.7f}, {self.velocity[2]:.7f} " \
                f"Orientation: {self.orientation[0]:.7f}, {self.orientation[1]:.7f}, {self.orientation[2]:.7f}"
        #return f"Velocity: {self.velocity[0]:.4f}, {self.velocity[1]:.4f}, {self.velocity[2]:.4f} "