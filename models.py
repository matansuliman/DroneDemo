from sensors import GPS, Rangefinder
from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicModel:
    def __init__(self, child_class_name, xml_name):
        self._xml_name = xml_name
        self._body_id = ENVIRONMENT.body_id(xml_name)
        self._sensors = dict()
        self._sensors['framepos'] = GPS(sensor_name= CONFIG[child_class_name]["sensors"]["framepos"])
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
        self._actuator_ids, self._actuator_names = ENVIRONMENT.actuators_for_body(self._body_id)
        LOGGER.info(f"\t\t\tModel: Initiated {self.__class__.__name__}")

    @property
    def actuator_ids(self):
        return self._actuator_ids

    @property
    def actuator_names(self):
        return self._actuator_names

    def get_height(self):
        return self._sensors['rangefinder'].get()

    def status(self):
        status = f"{self.__class__.__name__} status:"
        #status += f"\ttruepos: {print_array_of_nums(self.get_true_pos())}\n"
        for sensor_name, sensor_obj in self._sensors.items():
            status += f"\t{sensor_name}: {print_array_of_nums(sensor_obj.get())}"
        status += "\n"
        return status


class Pad(BasicModel):
    def __init__(self):
        super().__init__(child_class_name= self.__class__.__name__, xml_name= CONFIG["Pad"]["xml_body_name"])
        self._radius = CONFIG["Pad"]["radius"]
        self._joint_x_name = 'Pad_joint_x'
        self._joint_y_name = 'Pad_joint_y'
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

    def status(self):
        status = f"{self.__class__.__name__} status:"
        #status += f"\ttruepos: {print_array_of_nums(self.get_true_pos())}\n"
        for sensor_name, sensor_obj in self._sensors.items():
            status += f"\t{sensor_name}: {print_array_of_nums(sensor_obj.get())}"
        status += "\n"
        return status

