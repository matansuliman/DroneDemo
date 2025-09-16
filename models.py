from sensors import GPS, IMU, Rangefinder, Touch

from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicModel:
    def __init__(self, child_class_name, xml_name):
        self._xml_name = xml_name
        self._body_id = ENVIRONMENT.body_id(xml_name)
        self._sensors = dict()
        self._sensors["gps"] = GPS(
            pos_sensor_name=CONFIG[child_class_name]["sensors"]["pos"],
            vel_sensor_name=CONFIG[child_class_name]["sensors"]["vel"],
        )
        self._log = {
            "Time (sec)": [],
            "x_true": [],
            "y_true": [],
            "z_true": [],
            "vx_true": [],
            "vy_true": [],
            "vz_true": [],
            "x": [],
            "y": [],
            "z": [],
            "vx": [],
            "vy": [],
            "vz": [],
        }
        ENVIRONMENT.set_body_cda(
            body=self._xml_name, cda=CONFIG[child_class_name]["cda"]
        )

    @property
    def body_id(self):
        return self._body_id

    @property
    def sensors(self):
        return self._sensors

    @property
    def log(self):
        return self._log

    def get_pos(self):
        return self.sensors["gps"].get_pos()

    def get_vel(self):
        return self.sensors["gps"].get_vel()

    def get_true_pos(self):
        return ENVIRONMENT.world_pos_of_body(self._body_id)

    def get_true_vel(self):
        return ENVIRONMENT.world_linvel_of_body(self._body_id)

    def update_log(self):
        t = ENVIRONMENT.data.time
        x_true, y_true, z_true = self.get_true_pos()
        vx_true, vy_true, vz_true = self.get_vel()
        x, y, z = self.get_pos()
        vx, vy, vz = self.get_vel()

        # Append values to the log
        self._log["Time (sec)"].append(t)

        self._log["x_true"].append(x_true)
        self._log["y_true"].append(y_true)
        self._log["z_true"].append(z_true)

        self._log["vx_true"].append(vx_true)
        self._log["vy_true"].append(vy_true)
        self._log["vz_true"].append(vz_true)

        self._log["x"].append(x)
        self._log["y"].append(y)
        self._log["z"].append(z)

        self._log["vx"].append(vx)
        self._log["vy"].append(vy)
        self._log["vz"].append(vz)

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")


class Quadrotor(BasicModel):
    def __init__(self):
        super().__init__(
            child_class_name=self.__class__.__name__,
            xml_name=CONFIG["Quadrotor"]["xml_body_name"],
        )
        self._sensors["rangefinder"] = Rangefinder(
            range_sensor_name=CONFIG["Quadrotor"]["sensors"]["rangefinder"]
        )
        self._sensors["imu"] = IMU(
            framequat_sensor_name=CONFIG["Quadrotor"]["sensors"]["framequat"],
            gyro_sensor_name=CONFIG["Quadrotor"]["sensors"]["gyro"],
            accelerometer_sensor_name=CONFIG["Quadrotor"]["sensors"]["accelerometer"],
        )

        self._log.update(
            {
                "rangefinder": [],
                "orientation_x (roll)": [],
                "orientation_y (pitch)": [],
                "orientation_z (yaw)": [],
                "gyro_x": [],
                "gyro_y": [],
                "gyro_z": [],
                "acceleration_x": [],
                "acceleration_y": [],
                "acceleration_z": [],
            }
        )

        self._actuator_ids, self._actuator_names = ENVIRONMENT.actuators_for_body(
            self._body_id
        )
        LOGGER.info(f"\t\t\tModel: Initiated {self.__class__.__name__}")

    @property
    def actuator_ids(self):
        return self._actuator_ids

    @property
    def actuator_names(self):
        return self._actuator_names

    def get_orientation(self):
        return self._sensors["imu"].get_orientation()

    def get_gyro(self):
        return self._sensors["imu"].get_gyro()

    def get_accelerometer(self):
        return self._sensors["imu"].get_accelerometer()

    def get_height(self):
        return self._sensors["rangefinder"].get()

    def update_log(self):
        super().update_log()
        h_rf = self.get_height()
        r, p, y = self.get_orientation()
        gx, gy, gz = self.get_gyro()
        accx, accy, accz = self.get_accelerometer()

        # Append values to the log
        self._log["rangefinder"].append(h_rf)

        self._log["orientation_x (roll)"].append(r)
        self._log["orientation_y (pitch)"].append(p)
        self._log["orientation_z (yaw)"].append(y)

        self._log["gyro_x"].append(gx)
        self._log["gyro_y"].append(gy)
        self._log["gyro_z"].append(gz)

        self._log["acceleration_x"].append(accx)
        self._log["acceleration_y"].append(accy)
        self._log["acceleration_z"].append(accz)

    def status(self):
        status = f"{self.__class__.__name__} \ttrue status:"
        status += f"\tpos: {print_for_gui(self.get_true_pos())}"
        status += f"\tvel: {print_for_gui(self.get_true_vel())}\n"
        for sensor_obj in self._sensors.values():
            status += f"\t\t{sensor_obj.status()}\n"
        return status


class Pad(BasicModel):
    def __init__(self):
        super().__init__(
            child_class_name=self.__class__.__name__,
            xml_name=CONFIG["Pad"]["xml_body_name"],
        )
        self._sensors["touch"] = Touch(
            touch_sensor_name=CONFIG["Pad"]["sensors"]["touch"]
        )
        self._log.update({"touch": []})

        self._radius = CONFIG["Pad"]["radius"]
        self._joint_x_name = "Pad_joint_x"
        self._joint_y_name = "Pad_joint_y"
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

    def get_touch_force(self):
        return self._sensors["touch"].get()

    def update_log(self):
        super().update_log()
        touch_force = self.get_touch_force()

        # Append values to the log
        self._log["touch"].append(touch_force)

    def status(self):
        status = f"{self.__class__.__name__}\t\ttrue status:"
        status += f"\tpos: {print_for_gui(self.get_true_pos())}"
        status += f"\tvel: {print_for_gui(self.get_true_vel())}\n"
        for sensor_obj in self._sensors.values():
            status += f"\t\t{sensor_obj.status()}\n"
        return status
