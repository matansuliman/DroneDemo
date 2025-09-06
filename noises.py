import numpy as np

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicNoise:
    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class GPSNoise(BasicNoise):
    def __init__(self, bias_stddev= 2):
        super().__init__()
        self._bias = np.clip(np.random.normal(0, bias_stddev, size=3), 1, 2)
        self._bias[2] *= 0 # no vertical bias

    def step(self):
        offset = self._bias
        scale = 1
        return offset, scale

class RangefinderNoise(BasicNoise):
    def __init__(self, bias_stddev= 0.0001):
        super().__init__()
        self._bias = np.random.normal(0, bias_stddev, size= 1)

    def step(self):
        offset = self._bias
        scale = 1
        return offset, scale