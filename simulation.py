import time
import mujoco

from orchestrators import basicOrchestrator


class basicSimulationRunner():
    """Owns the simulation loop lifecycle (thread target).

    Keeps threading and loop orchestration out of app.py.
    """
    def __init__(self, orchestrator= basicOrchestrator(), loop_state= 'resume'):
        
        self._orchestrator = orchestrator
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
    
    def ChangeLoopState(self, terminate=False, pause=False, resume=False):
        raise NotImplementedError("Subclasses should implement this method")
    
    def isLoopState(self, state):
        raise NotImplementedError("Subclasses should implement this method")
    
    def run(self):
        raise NotImplementedError("Subclasses should implement this method")


PAUSE_SLEEP_SEC = 0.1

class SimulationRunner(basicSimulationRunner):
    def __init__(self, orchestrator):
        super().__init__(
            orchestrator= orchestrator,
            loop_state= 'resume'
            )
    
    def ChangeLoopState(self, terminate=False, pause=False, resume=False):
        if terminate:
            self._loop_state = 'terminate'
        elif pause:
            self._loop_state = 'pause'
        elif resume:
            self._loop_state = 'resume'
        else:
            raise ValueError("Invalid loop state change request.")
        
    def isLoopState(self, state):
        return self._loop_state == state

    def run(self):
        while True:
            mujoco.mj_step(self._env.model, self._env.data)
            
            if   self.isLoopState('terminate'): break
            elif self.isLoopState('pause'):     time.sleep(PAUSE_SLEEP_SEC)
            else:
                self._orchestrator.step_scene()
                time.sleep(self._env.dt)