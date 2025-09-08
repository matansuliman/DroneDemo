import time, threading

from PySide6.QtCore import Signal, QObject

from environment import ENVIRONMENT
from logger import LOGGER
from config import CONFIG


class BasicSimulationRunner:
    def __init__(self, orchestrator, loop_state):
        LOGGER.info("\tSimulation: Initiating")
        self._orchestrator = orchestrator()
        self._loop_state = loop_state

    @property
    def orchestrator(self):
        return self._orchestrator

    @property
    def loop_state(self):
        return self._loop_state
        
    def is_loop_state(self, state):
        return self._loop_state == state

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")


class SimulationRunner(QObject, BasicSimulationRunner):
    status_ready = Signal(str)

    def __init__(self, orchestrator, loop_state= 'resume'):
        super().__init__(orchestrator= orchestrator, loop_state= loop_state)
        self._pause_event = threading.Event()
        self._pause_event.clear() if loop_state == 'pause' else self._pause_event.set()
        LOGGER.info(f"\tSimulation: Initiated {self.__class__.__name__}")

    def terminate(self):
        self._loop_state = 'terminate'
        LOGGER.debug(f"Simulation: Terminate")

    def pause(self):
        self._loop_state = 'pause'
        self._pause_event.clear()
        LOGGER.debug("Simulation: Pause")

    def resume(self):
        self._loop_state = 'resume'
        self._pause_event.set()
        LOGGER.debug("Simulation: Resume")

    def status(self):
        status = "" #f"{self.__class__.__name__} status:\n"
        status += self.orchestrator.status()
        return status

    def _continue(self):
        return (not self.is_loop_state('terminate') and
               self._pause_event.wait())

    def run(self):
        LOGGER.debug("Simulation: Running")

        while self._continue():
            self._orchestrator.step_scene()  # advance scene
            self.status_ready.emit(self.status()) # emit status
            ENVIRONMENT.step()  # advance physics
            time.sleep(ENVIRONMENT.dt)  # keep real-time pacing

        LOGGER.info("Simulation: Terminated")