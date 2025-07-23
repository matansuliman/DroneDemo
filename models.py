from sensors import GPS, INS

XML_DRONE_NAME = 'x2'
XML_ACCEL_SENSOR_NAME = 'body_linacc'
XML_GYRO_SENSOR_NAME = 'body_gyro'


class Drone:
    def __init__(self, env, type='quadrotor'):
        self._type = type
        self._env = env
        self._bodyId = next(i for i in range(env.model.nbody) if env.model.body(i).name == XML_DRONE_NAME)

        self._motorMap = self._init_motor_map()
        self._sensors = self._init_sensors()
    
    def _init_motor_map(self):
        # Map motor actuator names to indices
        motor_names = [self._env.model.actuator(i).name for i in range(self._env.model.nu)]
        return {name: i for i, name in enumerate(motor_names)}
    
    def _init_sensors(self):
        sensors = {}
        sensors['gps'] = GPS(self._env, self._bodyId)
        sensors['ins'] = INS(self._env, self._bodyId)
        sensors['accel_sensor_id'] = next(i for i in range(self._env.model.nsensor) if self._env.model.sensor(i).name == XML_ACCEL_SENSOR_NAME)
        sensors['gyro_sensor_id'] = next(i for i in range(self._env.model.nsensor) if self._env.model.sensor(i).name == XML_GYRO_SENSOR_NAME)
        return sensors

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def bodyId(self):
        return self._bodyId
    
    @property
    def motorMap(self):
        return self._motorMap
    
    def getTruePos(self):
        return self.sensors['gps'].getTruePos()



import numpy as np
from sensors import GPS, INS

XML_NAME = 'platform'
XML_JOINT_NAME_X = 'platform_x'
XML_JOINT_NAME_Y = 'platform_y'

DEFAULT_VELOCITY = [0.0, 0.0, 0.0]


class MovingPlatform:
    def __init__(self, env, type= 'moving_platform', velocity=DEFAULT_VELOCITY):
        self._type = type
        self._env = env
        self._bodyId = next(i for i in range(env.model.nbody) if env.model.body(i).name == XML_NAME)

        self._sensors = self._init_sensors()

        self._velocity = np.array(velocity, dtype=np.float64)

        self._joint_x_id = env.model.joint(XML_JOINT_NAME_X).qposadr
        self._joint_y_id = env.model.joint(XML_JOINT_NAME_Y).qposadr

    def _init_sensors(self):
            sensors = {}
            sensors['gps'] = GPS(self._env, self._bodyId)
            sensors['ins'] = INS(self._env, self._bodyId)
            return sensors

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def bodyId(self):
        return self._bodyId

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

    def getTruePos(self):
        return self.sensors['gps'].getTruePos()

