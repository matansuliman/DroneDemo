from sensors import GPS

from logger import LOGGER
from config import CONFIG


class BasicModel:
    def __init__(self, env, xml_name):
        self._env = env
        self._xml_name = xml_name
        self._body_id = env.body_id(xml_name)
        self._sensors = {'gps': GPS(self._env, self._body_id)}

    @property
    def xml_name(self):
        return self._xml_name

    @property
    def body_id(self):
        return self._body_id

    @property
    def sensors(self):
        return self._sensors

    def get_pos(self, mode='noise'):
        return self.sensors['gps'].get_pos(mode=mode)


class Quadrotor(BasicModel):
    def __init__(self, env):
        super().__init__(env=env, xml_name= CONFIG["Quadrotor"]["xml_body_name"])
        self._env.set_body_cda(body= self._xml_name, cda= CONFIG["Quadrotor"]["cda"])
        LOGGER.info(f"\t\t\tModel: Initiated {self.__class__.__name__}")


class MovingPlatform(BasicModel):
    def __init__(self, env):
        super().__init__(env=env, xml_name= CONFIG["MovingPlatform"]["xml_body_name"])
        self._env.set_body_cda(body= self._xml_name, cda= CONFIG["MovingPlatform"]["cda"])
        self._radius = CONFIG["MovingPlatform"]["radius"]
        self._joint_x_name = 'platform_joint_x'
        self._joint_y_name = 'platform_joint_y'
        self._locks_end_pos = None
        self._locks_arms_length = CONFIG["MovingPlatform"]["locks_arms_length"]
        LOGGER.info(f"\t\t\tModel: Initiated {self.__class__.__name__}")

    @property
    def radius(self):
        return self._radius

    @property
    def joint_x_name(self):
        return self._joint_x_name

    @property
    def joint_y_name(self):
        return self._joint_y_name

    @property
    def locks_end_pos(self):
        return self._locks_end_pos

    @locks_end_pos.setter
    def locks_end_pos(self, value):
        self._locks_end_pos = value

    @property
    def locks_arms_length(self):
        return self._locks_arms_length