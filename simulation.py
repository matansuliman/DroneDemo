import time, threading

from logger import LOGGER
from config import CONFIG


class BasicSimulationRunner:
    def __init__(self, orchestrator, loop_state= 'pause'):
        LOGGER.info("\tSimulation: Initiating")
        self._orchestrator = orchestrator()
        self._env = self._orchestrator.env
        self._loop_state = loop_state

    @property
    def orchestrator(self):
        return self._orchestrator

    @property
    def env(self):
        return self._env

    @property
    def loop_state(self):
        return self._loop_state
    
    def set_loop_state(self, terminate=False, pause=False, resume=False):
        if terminate:
            self._loop_state = 'terminate'
            LOGGER.debug(f"Simulation: set_loop_state to terminate")
        elif pause:
            self._loop_state = 'pause'
            LOGGER.debug(f"Simulation: set_loop_state to pause")
        elif resume:
            self._loop_state = 'resume'
            LOGGER.debug(f"Simulation: set_loop_state to resume")
        else:
            LOGGER.error(f"Simulation: Invalid loop state change request!")
        
    def is_loop_state(self, state):
        return self._loop_state == state

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")


class SimulationRunner(BasicSimulationRunner):
    def __init__(self, orchestrator, loop_state= 'resume'):
        super().__init__(orchestrator= orchestrator, loop_state= loop_state)
        self._pause_event = threading.Event()
        LOGGER.info(f"\tSimulation: Initiated {self.__class__.__name__}")

    def set_loop_state(self, terminate=False, pause=False, resume=False):
        super().set_loop_state(terminate=terminate, pause=pause, resume=resume)
        if pause:
            self._pause_event.clear()
            LOGGER.debug("Simulation: Paused")
        elif resume:
            self._pause_event.set()
            LOGGER.debug("Simulation: Resuming")

    def run(self):
        LOGGER.debug("Simulation: Running")
        while not self.is_loop_state('terminate'):
            self._pause_event.wait()

            self._orchestrator.step_scene()  # advance scene
            self._env.step()  # advance physics
            time.sleep(self._env.dt)  # keep real-time pacing

            if self._orchestrator.scene_ended:
                self.set_loop_state(pause=True)

        LOGGER.info("Simulation: Terminated")