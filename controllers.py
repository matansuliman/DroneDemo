import numpy as np
from simple_pid import PID

from helpers import *

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicController:
    def __init__(self, plant):
        self._plant = plant

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")


class QuadrotorController(BasicController):

    def __init__(self, quadrotor):
        super().__init__(plant=quadrotor)

        conf_pid = CONFIG["QuadrotorController"]["pids"]
        self._pids = {
            "x": PID(
                Kp=conf_pid["x"]["kp"],
                Kd=conf_pid["x"]["kd"],
                setpoint=conf_pid["x"]["setpoint"],
                output_limits=sym_limits(conf_pid["x"]["output_limit"]),
            ),
            "y": PID(
                Kp=conf_pid["y"]["kp"],
                Kd=conf_pid["y"]["kd"],
                setpoint=conf_pid["y"]["setpoint"],
                output_limits=sym_limits(conf_pid["y"]["output_limit"]),
            ),
            "z": PID(
                Kp=conf_pid["z"]["kp"],
                Kd=conf_pid["z"]["kd"],
                setpoint=conf_pid["z"]["setpoint"],
                output_limits=sym_limits(conf_pid["z"]["output_limit"]),
            ),
            "roll": PID(
                Kp=conf_pid["ro"]["kp"],
                Kd=conf_pid["ro"]["kd"],
                setpoint=conf_pid["ro"]["setpoint"],
                output_limits=sym_limits(conf_pid["ro"]["output_limit"]),
            ),
            "pitch": PID(
                Kp=conf_pid["pi"]["kp"],
                Kd=conf_pid["pi"]["kd"],
                setpoint=conf_pid["pi"]["setpoint"],
                output_limits=sym_limits(conf_pid["pi"]["output_limit"]),
            ),
            "yaw": PID(
                Kp=conf_pid["ya"]["kp"],
                Kd=conf_pid["ya"]["kd"],
                setpoint=conf_pid["ya"]["setpoint"],
                output_limits=sym_limits(conf_pid["ya"]["output_limit"]),
            ),
        }

        conf_ff = CONFIG["QuadrotorController"]["ff"]
        self._ff = {
            "x": conf_ff["x"],
            "y": conf_ff["y"],
            "z": conf_ff["z"],
            "throttle": conf_ff["throttle"],
        }

        self._reference = None
        self._turned_off = False
        self._descend = False
        descend_phases = CONFIG["QuadrotorController"]["descend_phases"]
        self._descend_phases = list(
            zip(
                descend_phases["names"],
                descend_phases["ff_z"],
                descend_phases["pid_z_ul"],
                descend_phases["pid_z_ll"],
                descend_phases["epsilon"],
            )
        )

        LOGGER.info(f"\t\t\tController: Initiated {self.__class__.__name__}")

    @property
    def descend(self):
        return self._descend

    @descend.setter
    def descend(self, value):
        self._descend = value

    def set_reference(self, pos, vel):
        self._reference = (pos, vel)
        # Update PID setpoints
        self._pids["x"].setpoint = pos[0] + self._ff["x"] * vel[0]
        self._pids["y"].setpoint = pos[1] + self._ff["y"] * vel[1]
        self._pids["z"].setpoint = pos[2] + self._ff["z"]

    def get_reference(self):
        return self._reference

    def _outer_loop(self):
        pos = self._plant.get_pos()

        # setpoints and cmd
        hz = CONFIG["QuadrotorController"]["outer_loop"]["hz"]
        interval = ENVIRONMENT.dt * hz
        current_time = ENVIRONMENT.get_time()
        previous_time = current_time - ENVIRONMENT.dt

        if (current_time // interval) != (previous_time // interval):
            self._pids["pitch"].setpoint = self._pids["x"](pos[0])
            self._pids["roll"].setpoint = -self._pids["y"](pos[1])

        # Altitude throttle
        throttle = self._pids["z"](pos[2]) + self._ff["throttle"]
        return throttle

    def _inner_loop(self):
        roll, pitch, yaw = self._plant.get_orientation()

        # Control signals
        roll_cmd = self._pids["roll"](roll)
        pitch_cmd = self._pids["pitch"](pitch)
        yaw_cmd = self._pids["yaw"](yaw)

        return roll_cmd, pitch_cmd, yaw_cmd

    def _apply_cmds(self, throttle, roll_cmd, pitch_cmd, yaw_cmd):
        # Motor Mixing Algorithm (MMA): Combine control signals to actuators
        M = np.array(CONFIG["QuadrotorController"]["mixing-matrix"])  # mixing matrix
        u = np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd])  # control vector
        thrusts = M @ u  # compute thrusts

        values = {
            "thrust1": thrusts[0],
            "thrust2": thrusts[1],
            "thrust3": thrusts[2],
            "thrust4": thrusts[3],
        }
        ENVIRONMENT.set_ctrls(values)

    def _get_phase(self):
        val = self._plant.sensors["rangefinder"].get()
        if val is None:
            return None
        for name, ff_z, pid_z_ul, pid_z_ll, epsilon in self._descend_phases:
            if val > ff_z + epsilon:
                return name, ff_z, pid_z_ul, pid_z_ll
        return "turn_off", 0, 0, 0

    def _get_phase_name(self):
        return self._get_phase()[0]

    def _enforce_descend(self):
        _, ff_z, pid_z_ul, pid_z_ll = self._get_phase()
        self._ff["z"] = ff_z
        self._pids["z"].output_limits = pid_z_ll, pid_z_ul

    def _enforce_hover(self):
        self._ff["z"] = CONFIG["QuadrotorController"]["ff"]["z"]
        self._pids["z"].output_limits = sym_limits(
            CONFIG["QuadrotorController"]["pids"]["z"]["output_limit"]
        )

    def _turn_off_plant(self):
        self._turned_off = True
        ENVIRONMENT.set_ctrls({name: 0.0 for name in self._plant.actuator_names})

    def status(self):
        status = f"{self.__class__.__name__} status:"
        status += f"\treference_pos: {print_for_gui(self.get_reference()[0])}"
        status += f"\treference_vel: {print_for_gui(self.get_reference()[1])}"
        if self._descend:
            status += f"\tdescend_phase: {self._get_phase()[0]}"
        status += "\n"
        return status

    def is_done(self):
        return self._get_phase_name() == "turn_off"

    def step(self):
        # Enforce turn off
        if self._turned_off:
            pass

        elif self.is_done():
            self._turn_off_plant()

        else:
            # Enforce descend or hover
            self._enforce_descend() if self._descend else self._enforce_hover()

            # Outer loop position control -> throttle
            # Inner loop orientation control -> (roll_cmd, pitch_cmd, yaw_cmd)
            # Apply control signals to actuators
            self._apply_cmds(self._outer_loop(), *self._inner_loop())


class PadController(BasicController):
    def __init__(self, pad):
        super().__init__(plant=pad)
        self._velocity = np.array(CONFIG["Pad"]["default_velocity"], dtype=np.float64)
        LOGGER.info(f"\t\t\tPadController: Initiated {self.__class__.__name__}")

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = np.array(velocity, dtype=np.float64)

    def status(self):
        status = f"{self.__class__.__name__} status:\t"
        status += f"\tvel: {print_for_gui(self.velocity)}\n"
        return status

    def step(self):
        # Update velocity
        ENVIRONMENT.set_joint_qvel(self._plant.joint_x_name, float(self._velocity[0]))
        ENVIRONMENT.set_joint_qvel(self._plant.joint_y_name, float(self._velocity[1]))
