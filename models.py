import numpy as np

from sensors import GPS, INS


class BasicModel:
    def __init__(self, env, name, type_str= 'basicModel'):
        self._env = env
        self._type_str = type_str
        self._bodyId = self._bodyId = env.body_id(name)
        self._sensors = None

    @property
    def type_str(self):
        return self._type_str

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def body_id(self):
        return self._bodyId
    
    def get_pos(self, mode='noise'):
        return self.sensors['gps'].get_pos(mode=mode)


XML_DRONE_NAME = 'x2'
XML_ACCEL_SENSOR_NAME = 'body_linacc'
XML_GYRO_SENSOR_NAME = 'body_gyro'


class Drone(BasicModel):
    def __init__(self, env, type_str='quadrotor'):
        super().__init__(env, XML_DRONE_NAME, type_str)
        self._sensors = {
            'gps': GPS(self._env, self._bodyId),
            'ins': INS(self._env, self._bodyId),
            'accel_sensor_id': self._env.sensor_id(XML_ACCEL_SENSOR_NAME),
            'gyro_sensor_id':  self._env.sensor_id(XML_GYRO_SENSOR_NAME),
        }
    
    @property
    def sensors(self):
        return self._sensors


XML_PLATFORM_NAME = 'platform'
XML_PLATFORM_JOINT_NAME_X = 'platform_x'
XML_PLATFORM_JOINT_NAME_Y = 'platform_y'

DEFAULT_VELOCITY = (0.0, 0.0, 0.0)


class MovingPlatform(BasicModel):
    def __init__(self, env, type_str= 'moving_platform', velocity=DEFAULT_VELOCITY):
        super().__init__(env, XML_PLATFORM_NAME, type_str)
        self._sensors = {
                'gps': GPS(self._env, self._bodyId),
                'ins': INS(self._env, self._bodyId)
            }
        self._velocity = np.array(velocity, dtype=np.float64)
        self._joint_x_id = env.model.joint(XML_PLATFORM_JOINT_NAME_X).qposadr
        self._joint_y_id = env.model.joint(XML_PLATFORM_JOINT_NAME_Y).qposadr

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def joint_x_id(self):
        return self._joint_x_id
    
    @property
    def joint_y_id(self):
        return self._joint_y_id