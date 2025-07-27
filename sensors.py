import numpy as np
from scipy.spatial.transform import Rotation as R
from noises import GPSNoise

class basicSensor():
    def __init__(self, env, bodyId: int):
        self._env = env
        self._bodyId = bodyId
    
    @property
    def bodyId(self):
        return self._bodyId
    
    def getPos(self, mode='noise'):
        raise NotImplementedError("Subclasses should implement this method")


class GPS(basicSensor):
    def __init__(self, env, bodyId: int):
        super().__init__(env, bodyId)
        self._noise = GPSNoise(self._env) # Initialize GPS noise model

    def getPos(self, mode='noise'):
        if mode == 'noise':
            offset, scale = self._noise.step()
            return (self.getPos(mode='no_noise') + offset) * scale
        
        elif mode == 'no_noise':
            return self._env.data.xpos[self._bodyId]
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    
class INS(basicSensor):
    def __init__(self, env, bodyId: int):
        super().__init__(env, bodyId)
        # Initialize position, velocity, and orientation
        self._position = self._env.data.xpos[self._bodyId]  # Initial position
        self._velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity
        self._orientation = np.array([0.0, 0.0, 0.0])  # Euler angles (roll, pitch, yaw)

    @property
    def position(self):
        return self._position
    
    @property
    def velocity(self):
        return self._velocity
    
    @property
    def orientation(self):
        return self._orientation
    
    def update(self, accel_data, gyro_data):
        self._orientation += gyro_data * self._env.dt # Update orientation

        r = R.from_euler('xyz', self._orientation) # Convert to rotation matrix
        accel_world = r.apply(accel_data) # Rotate acceleration to world frame

        # Integrate
        self._velocity += accel_world * self._env.dt
        self._position += self._velocity * self._env.dt
    
    def __str__(self):
        return f"Position: {self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f} " \
                f"Vellocity: {self.velocity[0]:.7f}, {self.velocity[1]:.7f}, {self.velocity[2]:.7f} " \
                f"Orientation: {self.orientation[0]:.7f}, {self.orientation[1]:.7f}, {self.orientation[2]:.7f}"
        #return f"Velocity: {self.velocity[0]:.4f}, {self.velocity[1]:.4f}, {self.velocity[2]:.4f} "