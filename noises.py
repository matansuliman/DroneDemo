import numpy as np

import logging
logger = logging.getLogger("app")


class BasicNoise:
    def __init__(self, env= None):
        self._env = env

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class GPSNoise(BasicNoise):
    def __init__(self, env, bias_stddev: float = 2):
        super().__init__(env= env)
        self.bias = np.random.normal(0, bias_stddev, size=3)
        self.bias[2] *= 0

    def step(self):
        offset = self.bias
        scale = 1
        return offset, scale