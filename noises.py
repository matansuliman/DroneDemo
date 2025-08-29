import numpy as np


class BasicNoise:
    def __init__(self, env):
        self._env = env

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def reset(self):
        raise NotImplementedError("Subclasses should implement this method")


DEFAULT_BIAS_STD = 1
DEFAULT_DRIFT_RATE_STD = 0
DEFAULT_SCALE_NOISE_STD = 0

Z_BIAS_SCALE = 0


class GPSNoise(BasicNoise):
    """
    GPSNoiseModel simulates GPS noise with:
    - static bias (fixed offset)
    - drift (accumulating over time)
    - multiplicative Gaussian noise (scaling factor)
    """

    def __init__(self, env,
                 bias_stddev: float = DEFAULT_BIAS_STD, 
                 drift_rate_stddev: float = DEFAULT_DRIFT_RATE_STD, 
                 scale_noise_stddev: float = DEFAULT_SCALE_NOISE_STD):
        
        super().__init__(env)
        self.bias = np.random.normal(0, bias_stddev, size=3)
        self.drift_rate = np.random.normal(0, drift_rate_stddev, size=3)
        self.drift = np.zeros(3)
        self.scale_noise_stddev = scale_noise_stddev

        self.bias[2] *= Z_BIAS_SCALE

    def step(self):
        self.drift += self.drift_rate * self._env.dt
        offset = self.bias + self.drift
        scale = np.random.normal(1.0, self.scale_noise_stddev, size=3)
        return offset, scale

    def reset(self):
        self.drift = np.zeros(3)