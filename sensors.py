from noises import GPSNoise, RangefinderNoise

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicSensor:
    def __init__(self, sensor_name):
        self._sensor_name = sensor_name
        self._sid = ENVIRONMENT.sensor_id(sensor_name)  # numeric sensor-id
        self._adr = ENVIRONMENT.model.sensor_adr[self._sid]  # start index into sensor-data
        self._dim = ENVIRONMENT.model.sensor_dim[self._sid]

    @property
    def sensor_name(self):
        return self._sensor_name

    @property
    def sid(self):
        return self._sid

    @property
    def adr(self):
        return self._adr

    @property
    def dim(self):
        return self._dim
    
    def get(self):
        return ENVIRONMENT.data.sensordata[self._adr : self._adr + self._dim]


class GPS(BasicSensor):
    def __init__(self, sensor_name):
        super().__init__(sensor_name)
        self._noise = GPSNoise()

    def get(self):
        offset, scale = self._noise.step()
        vals = ENVIRONMENT.data.sensordata[self._adr : self._adr + self._dim]
        return (vals + offset) * scale

class Rangefinder(BasicSensor):
    def __init__(self, sensor_name):
        super().__init__(sensor_name)
        self._noise = RangefinderNoise()

    def get(self):
        offset, scale = self._noise.step()
        val = ENVIRONMENT.data.sensordata[self._adr : self._adr + self._dim][0]
        return None if val < 0 else (val + offset) * scale
