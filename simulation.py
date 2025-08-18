import time

from orchestrators import BasicOrchestrator, FollowTarget

class BasicSimulationRunner:
    """Owns the simulation loop lifecycle (thread target).

    Keeps threading and loop orchestration out of app.py.
    """
    def __init__(self, orchestrator= BasicOrchestrator, loop_state= 'resume'):
        
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
        elif pause:
            self._loop_state = 'pause'
        elif resume:
            self._loop_state = 'resume'
        else:
            raise ValueError("Invalid loop state change request.")
        
    def is_loop_state(self, state):
        return self._loop_state == state

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")


PAUSE_SLEEP_SEC = 0.1

class SimulationRunner(BasicSimulationRunner):
    def __init__(self, orchestrator= BasicOrchestrator, loop_state= 'resume'):
        super().__init__(
            orchestrator= orchestrator,
            loop_state= loop_state
            )

    def run(self):
        while True:
            if self.is_loop_state('terminate'): break
            if self.is_loop_state('pause'):
                time.sleep(PAUSE_SLEEP_SEC)
                continue
            
            self._orchestrator.step_scene() # compute scene logic & write controls
            self._env.step()                # advance physics with our effects (wind/drag etc.)
            time.sleep(self._env.dt)        # keep real-time pacing