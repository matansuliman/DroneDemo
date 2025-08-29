from sensors import GPS


class BasicModel:
    def __init__(self, info: str ='', env= None, xml_name= ''):
        self._info = info
        self._env = env
        self._body_id = env.body_id(xml_name)
        self._sensors = None

    @property
    def info(self):
        return self._info

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def body_id(self):
        return self._body_id
    
    def get_pos(self, mode='noise'):
        return self.sensors['gps'].get_pos(mode=mode)

    def __str__(self):
        return f'model ({self.__class__.__name__}) info: {self._info}'


XML_DRONE_NAME = 'x2'
XML_ACCEL_SENSOR_NAME = 'body_linac'
XML_GYRO_SENSOR_NAME = 'body_gyro'


class Quadrotor(BasicModel):
    def __init__(self, info: str='quadrotor', env= None, xml_name: str= XML_DRONE_NAME):
        super().__init__(info=info, env=env, xml_name=xml_name)
        self._sensors = {
            'gps': GPS(self._env, self._body_id),
            'accel_sensor_id': self._env.sensor_id(XML_ACCEL_SENSOR_NAME),
            'gyro_sensor_id':  self._env.sensor_id(XML_GYRO_SENSOR_NAME),
        }
    
    @property
    def sensors(self):
        return self._sensors


XML_PLATFORM_NAME = 'platform'
XML_PLATFORM_JOINT_NAME_X = 'platform_x'
XML_PLATFORM_JOINT_NAME_Y = 'platform_y'

class MovingPlatform(BasicModel):
    def __init__(self, info: str = 'moving_platform', env=None, xml_name: str = XML_PLATFORM_NAME):
        super().__init__(info=info, env=env, xml_name=xml_name)
        self._sensors = {'gps': GPS(self._env, self._body_id)}

        self._joint_x_name = "platform_x"
        self._joint_y_name = "platform_y"

    @property
    def joint_x_name(self):
        return self._joint_x_name

    @property
    def joint_y_name(self):
        return self._joint_y_name