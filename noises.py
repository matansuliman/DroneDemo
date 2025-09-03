import numpy as np

from logger import LOGGER
from config import CONFIG



class BasicNoise:
    def __init__(self, env):
        self._env = env

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class GPSNoise(BasicNoise):
    def __init__(self, env, bias_stddev= 2):
        super().__init__(env= env)
        self.bias = np.random.normal(0, bias_stddev, size=3)
        self.bias[2] *= 0 # no vertical bias

    def step(self):
        offset = self.bias
        scale = 1
        return offset, scale