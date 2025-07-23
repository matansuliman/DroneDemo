DEFAULT_DT = 0.01

class ENV:
    def __init__(self, model, data, dt :float = DEFAULT_DT):
        self._model = model
        self._data = data
        self._dt = dt
    
    @property
    def model(self):
        return self._model
    
    @property
    def data(self):
        return self._data
    
    @property
    def dt(self):
        return self._dt
    
    def getTime(self):
        return self._data.time
