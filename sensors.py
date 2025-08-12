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
                f"Velocity: {self.velocity[0]:.7f}, {self.velocity[1]:.7f}, {self.velocity[2]:.7f} " \
                f"Orientation: {self.orientation[0]:.7f}, {self.orientation[1]:.7f}, {self.orientation[2]:.7f}"


class FusionFilter:
    """
    Sensor fusion filter for combining INS and GPS readings.

    Default model: complementary filter
    Default params: {'alpha': 0.98}
    """
    def __init__(self, model: str = 'complementary', params: dict = {}):
        self.model = model
        self.params = {'alpha': 0.98} if model == 'complementary' else params
        
        # internal state
        self._fused_pos = np.zeros(3)
        self._fused_vel = np.zeros(3)
        self._last_time = None

    @property
    def position(self):
        """Get current fused position."""
        return self._fused_pos.copy()

    @property
    def velocity(self):
        """Get current fused velocity."""
        return self._fused_vel.copy()

    def reset(self, initial_pos = [0,0,0], initial_vel = [0,0,0]):
        """Reset internal state to known values."""
        self._fused_pos = np.array(initial_pos, dtype=float)
        self._fused_vel = np.array(initial_vel, dtype=float)
        self._last_time = None

    def update(self, ins_pos: np.ndarray, ins_vel: np.ndarray,
               gps_pos: np.ndarray, timestamp: float):
        """
        Update fused state given:
          - ins_pos, ins_vel : INS dead-reckoned position/velocity
          - gps_pos           : noisy GPS measurement
          - timestamp         : current time (s)
        """
        if self.model == 'complementary':
            alpha = self.params['alpha']
            if self._last_time is None:
                # first call: initialize
                self._fused_pos = gps_pos.copy()
                self._fused_vel = ins_vel.copy()
            else:
                dt = timestamp - self._last_time
                # predict step: propagate INS estimate
                pred_pos = self._fused_pos + ins_vel * dt
                # correct step: blend with GPS
                self._fused_pos = alpha * pred_pos + (1 - alpha) * gps_pos
                self._fused_vel = ins_vel  # trust INS for velocity
        else:
            raise NotImplementedError(f"Fusion model '{self.model}' is not implemented.")

        self._last_time = timestamp
        return self._fused_pos, self._fused_vel

    
