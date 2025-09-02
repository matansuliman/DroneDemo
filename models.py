from sensors import GPS

import logging
logger = logging.getLogger("app")


class BasicModel:
    def __init__(self, info: str ='', env= None, xml_name= ''):
        self._info = info
        self._env = env
        self._body_id = env.body_id(xml_name)
        self._sensors = {'gps': GPS(self._env, self._body_id)}

    @property
    def info(self):
        return self._info

    @property
    def body_id(self):
        return self._body_id

    @property
    def sensors(self):
        return self._sensors

    def get_pos(self, mode='noise'):
        return self.sensors['gps'].get_pos(mode=mode)

    def __str__(self):
        return f'model ({self.__class__.__name__}) info: {self._info}'


class Quadrotor(BasicModel):
    XML_BODY_NAME = 'x2'
    CDA = 0.04

    def __init__(self, info: str='quadrotor', env= None):
        super().__init__(info=info, env=env, xml_name=Quadrotor.XML_BODY_NAME)
        self._env.set_body_cda(body=Quadrotor.XML_BODY_NAME, cda=Quadrotor.CDA)


class MovingPlatform(BasicModel):
    XML_BODY_NAME = 'platform_body'
    CDA = 0.25

    def __init__(self, info: str = 'moving_platform', env=None):
        super().__init__(info=info, env=env, xml_name=MovingPlatform.XML_BODY_NAME)
        self._env.set_body_cda(body=MovingPlatform.XML_BODY_NAME, cda=MovingPlatform.CDA)
        self._radius = 1
        self._joint_x_name = 'platform_joint_x'
        self._joint_y_name = 'platform_joint_y'

    @property
    def radius(self):
        return self._radius

    @property
    def joint_x_name(self):
        return self._joint_x_name

    @property
    def joint_y_name(self):
        return self._joint_y_name