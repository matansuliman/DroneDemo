import time

from orchestrators import BasicOrchestrator

from logger_config import log_multiline
import logging
logger = logging.getLogger("app")


class BasicSimulationRunner:
    def __init__(self, info: str= '', orchestrator= BasicOrchestrator, loop_state= 'pause'):
        self._info = info
        self._orchestrator = orchestrator()
        self._env = self._orchestrator.env
        self._loop_state = loop_state
        log_multiline(logger, str(self))

    @property
    def info(self):
        return self._info

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
            logger.debug(f"Simulation: set_loop_state to terminate")
        elif pause:
            self._loop_state = 'pause'
            logger.debug(f"Simulation: set_loop_state to pause")
        elif resume:
            self._loop_state = 'resume'
            logger.debug(f"Simulation: set_loop_state to resume")
        else:
            logger.error(f"Simulation: Invalid loop state change request!")
        
    def is_loop_state(self, state):
        return self._loop_state == state

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __str__(self):
        return f'simulation ({self.__class__.__name__}) info: {self._info}\n\t{self._orchestrator}'


import threading

class SimulationRunner(BasicSimulationRunner):
    def __init__(self, info: str= 'runs mujoco', orchestrator= BasicOrchestrator, loop_state= 'resume'):
        super().__init__(info = info, orchestrator= orchestrator, loop_state= loop_state )
        self._pause_event = threading.Event()

    def set_loop_state(self, terminate=False, pause=False, resume=False):
        super().set_loop_state(terminate=terminate, pause=pause, resume=resume)
        if pause: self._pause_event.clear()
        elif resume: self._pause_event.set()

    def run(self):
        while not self.is_loop_state('terminate'):
            self._pause_event.wait()

            self._orchestrator.step_scene() # compute scene logic & write controls
            self._env.step()                # advance physics with our effects (wind/drag etc.)
            time.sleep(self._env.dt)        # keep real-time pacing