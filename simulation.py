import time

PAUSE_SLEEP_SEC = 0.1


class SimulationRunner:

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator

    def run(self):
        
        while True:
            self._orchestrator.mujuco_step()

            if self._orchestrator.loop_terminated: 
                break

            elif self._orchestrator.loop_paused:
                time.sleep(PAUSE_SLEEP_SEC)
                continue
            
            else:
                self._orchestrator.update_scene()
                time.sleep(self._orchestrator._env.dt)