import numpy as np

from sensors import GPS, INS


class basicModel:
    def __init__(self, env, name, type= 'basicModel'):
        self._env = env
        self._type = type
        self._bodyId = next(i for i in range(env.model.nbody) if env.model.body(i).name == name)

    @property
    def type(self):
        return self._type

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def bodyId(self):
        return self._bodyId
    
    def getPos(self, mode='noise'):
        return self.sensors['gps'].getPos(mode=mode)


XML_DRONE_NAME = 'x2'
XML_ACCEL_SENSOR_NAME = 'body_linacc'
XML_GYRO_SENSOR_NAME = 'body_gyro'


class Drone(basicModel):
    def __init__(self, env, type='quadrotor'):
        super().__init__(env, XML_DRONE_NAME, type)
        self._sensors = {
            'gps': GPS(self._env, self._bodyId),
            'ins': INS(self._env, self._bodyId),
            'accel_sensor_id': next(i for i in range(self._env.model.nsensor) if self._env.model.sensor(i).name == XML_ACCEL_SENSOR_NAME),
            'gyro_sensor_id': next(i for i in range(self._env.model.nsensor) if self._env.model.sensor(i).name == XML_GYRO_SENSOR_NAME)
        }

        # Map motor actuator names to indices
        _motor_names = [self._env.model.actuator(i).name for i in range(self._env.model.nu)]
        self._motorMap = {name: i for i, name in enumerate(_motor_names)}
    
    @property
    def sensors(self):
        return self._sensors
    
    @property
    def motorMap(self):
        return self._motorMap





XML_PLATFORM_NAME = 'platform'
XML_PLATFORM_JOINT_NAME_X = 'platform_x'
XML_PLATFORM_JOINT_NAME_Y = 'platform_y'

DEFAULT_VELOCITY = [0.0, 0.0, 0.0]


class MovingPlatform(basicModel):
    def __init__(self, env, type= 'moving_platform', velocity=DEFAULT_VELOCITY):
        super().__init__(env, XML_PLATFORM_NAME, type)
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