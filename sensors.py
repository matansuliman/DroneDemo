from noises import *
from helpers import *

from scipy.spatial.transform import Rotation

from environment import ENVIRONMENT


class MujocoSensor:
    def __init__(self, sensor_name):
        self._sensor_name = sensor_name
        self._sid = ENVIRONMENT.sensor_id(sensor_name)  # numeric sensor-id
        self._adr = ENVIRONMENT.model.sensor_adr[
            self._sid
        ]  # start index into sensor-data
        self._dim = ENVIRONMENT.model.sensor_dim[self._sid]

    @property
    def sensor_name(self):
        return self._sensor_name

    def get(self):
        return ENVIRONMENT.data.sensordata[self._adr : self._adr + self._dim]

    def status(self):
        status = "Basic Sensor status:"
        status += f"\tget: {self.get()}"
        return status


class GPS:
    def __init__(self, pos_sensor_name, vel_sensor_name):
        self._pos_sensor = MujocoSensor(sensor_name=pos_sensor_name)
        self._vel_sensor = MujocoSensor(sensor_name=vel_sensor_name)
        self._pos_noise = PosNoise()
        self._vel_noise = VelNoise()

    def get(self):
        return self.get_pos(), self.get_vel()

    def get_pos(self):
        vals = self._pos_sensor.get()
        vals_noised = self._pos_noise.apply(vals)
        return vals_noised

    def get_vel(self):
        vals = self._vel_sensor.get()
        vals_noised = self._vel_noise.apply(vals)
        return vals_noised

    def status(self):
        status = f"{self.__class__.__name__} status:"
        status += f"\tpos: {print_for_gui(self.get_pos())}"
        status += f"\tvel: {print_for_gui(self.get_vel())}"
        status += f"\tpos noise: {self._pos_noise}"
        return status


class IMU:
    def __init__(
        self, framequat_sensor_name, gyro_sensor_name, accelerometer_sensor_name
    ):
        self._quat_sensor = MujocoSensor(sensor_name=framequat_sensor_name)
        self._gyro_sensor = MujocoSensor(sensor_name=gyro_sensor_name)
        self._accelerometer_sensor = MujocoSensor(sensor_name=accelerometer_sensor_name)

        self._orientation_noise = QuatNoise()
        self._gyro_noise = GyroNoise()
        self._accelerometer_noise = AccelerometerNoise()

    def get(self):
        return self.get_quat_wxyz(), self.get_gyro(), self.get_accelerometer()

    def get_quat_wxyz(self):
        return self._quat_sensor.get()

    def get_orientation(self):
        return Rotation.from_quat(self.get_quat_wxyz()[[1, 2, 3, 0]]).as_euler(
            "xyz", degrees=False
        )

    def get_gyro(self):
        omega = np.array(self._gyro_sensor.get(), dtype=float)
        omega = self._gyro_noise.apply(omega)
        return omega

    def get_accelerometer(self):
        accelerometer = np.array(self._accelerometer_sensor.get(), dtype=float)
        accelerometer = self._accelerometer_noise.apply(accelerometer)
        return accelerometer

    def status(self):
        quat_wxyz, omega, accelerometer = self.get()
        orientation = self.get_orientation()
        return (
            f"{self.__class__.__name__} status:"
            f"\tquat_wxyz: {print_for_gui(quat_wxyz)}"
            f"\torientation: {print_for_gui(orientation)}"
            f"\tgyro(rad/s): {print_for_gui(omega)}"
            f"\taccelerometer(m/s^2): {print_for_gui(accelerometer)}"
        )


class Rangefinder:
    def __init__(self, range_sensor_name):
        self._range_sensor = MujocoSensor(sensor_name=range_sensor_name)
        self._range_noise = RangefinderNoise()

    def get(self):
        range_val = self._range_sensor.get()[0]
        range_val_noised = self._range_noise.apply(range_val)
        return None if range_val < 0 else range_val_noised

    def status(self):
        status = f"{self.__class__.__name__} status:"
        status += f"\tpos: {print_for_gui(self.get())}"
        status += f"\tnoise: {self._range_noise}"
        return status


class Touch:
    def __init__(self, touch_sensor_name: str):
        self._sensor = MujocoSensor(sensor_name=touch_sensor_name)

    def get(self):
        vals = self._sensor.get()
        return float(vals[0]) if len(vals) == 1 else vals

    def status(self):
        return f"Touch forces:{self.get()}"
