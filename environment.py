import mujoco

PATH_TO_XML = "skydio_x2/scene.xml"

class ENV:
    def __init__(self, path_to_xml= PATH_TO_XML):
        self._model = mujoco.MjModel.from_xml_path(path_to_xml)
        self._data= mujoco.MjData(self._model)
        self._dt= self._model.opt.timestep
    
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
