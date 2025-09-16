import numpy as np

from helpers import *

from config import CONFIG


class BasicNoise:
    def __init__(self, scale=1, offset=1, drift_rate=0):
        self._scale = scale
        self._offset = offset
        self._drift_rate = drift_rate
        self._drift = 0

    def _step(self):
        self._drift += self._drift_rate

    def apply(self, val):
        self._step()
        return (val + self._offset + self._drift) * self._scale

    def __str__(self):
        res = ""
        if self._scale != 1:
            res += f"scale: {self._scale}"
        res += f"offset: {print_for_gui(self._offset)}"
        if self._drift_rate != 0:
            res += f"drift_rate: {self._drift_rate}, drift: {self._drift}"
        return res


class PosNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Pos"]
        super().__init__(
            scale=conf["scale"],
            offset=np.append(
                generate_normal_clipped(*conf["offset"].values(), size=2), 0
            ),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")


class VelNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Vel"]
        super().__init__(
            scale=conf["scale"],
            offset=np.append(
                generate_normal_clipped(*conf["offset"].values(), size=2), 0
            ),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")


class QuatNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Quat"]
        super().__init__(
            scale=conf["scale"],
            offset=generate_normal_clipped(*conf["offset"].values(), size=4),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")


class GyroNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Gyro"]
        super().__init__(
            scale=conf["scale"],
            offset=generate_normal_clipped(*conf["offset"].values(), size=3),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")


class AccelerometerNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Accelerometer"]
        super().__init__(
            scale=conf["scale"],
            offset=generate_normal_clipped(*conf["offset"].values(), size=3),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")


class RangefinderNoise(BasicNoise):
    def __init__(self):
        conf = CONFIG["Noises"]["Rangefinder"]
        super().__init__(
            scale=conf["scale"],
            offset=generate_normal_clipped(*conf["offset"].values(), size=1),
            drift_rate=conf["drift_rate"],
        )
        # LOGGER.debug(f"{self.__class__.__name__} created: {self}")
