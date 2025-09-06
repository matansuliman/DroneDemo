from sensors import GPS, Rangefinder
from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicModel:
    def __init__(self, child_class_name, xml_name):
        self._xml_name = xml_name
        self._body_id = ENVIRONMENT.body_id(xml_name)
        self._sensors = {'framepos': GPS(sensor_name= CONFIG[child_class_name]["sensors"]["framepos"])}
        ENVIRONMENT.set_body_cda(body= self._xml_name, cda= CONFIG[child_class_name]["cda"])

    @property
    def xml_name(self):
        return self._xml_name

    @property
    def body_id(self):
        return self._body_id

    @property
    def sensors(self):
        return self._sensors

    def get_pos(self):
        return self.sensors['framepos'].get()

    def get_true_pos(self):
        return ENVIRONMENT.world_pos_of_body(self._body_id)

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")


class Quadrotor(BasicModel):
    def __init__(self):
        super().__init__(child_class_name= self.__class__.__name__, xml_name= CONFIG["Quadrotor"]["xml_body_name"])
        self._sensors["rangefinder"] = Rangefinder(sensor_name= CONFIG["Quadrotor"]["sensors"]["rangefinder"])
        LOGGER.info(f"\t\t\tModel: Initiated {self.__class__.__name__}")

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"\ttruepos: {print_array_of_nums(self.get_true_pos())}\n"
        status += f"\tframepos: {print_array_of_nums(self.get_pos())}\n"
        status += f"\trangefinder: {print_array_of_nums(self.sensors['rangefinder'].get())}\n"
        return status


class Pad(BasicModel):
    def __init__(self):
        super().__init__(child_class_name= self.__class__.__name__, xml_name= CONFIG["Pad"]["xml_body_name"])
        self._radius = CONFIG["Pad"]["radius"]
        self._joint_x_name = 'Pad_joint_x'
        self._joint_y_name = 'Pad_joint_y'
        self._locks_end_pos = None
        self._locks_arms_length = CONFIG["Pad"]["locks_arms_length"]
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

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"\ttruepos: {print_array_of_nums(self.get_true_pos())}\n"
        status += f"\tframepos: {print_array_of_nums(self.get_pos())}\n"
        return status

