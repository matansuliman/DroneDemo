class SimulationWrapper:

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator

    def run(self):
        self._orchestrator.loop()