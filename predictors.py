

from collections import deque
import cv2
import numpy as np

class basicDetector():
    def __init__(self,name = "", model = None, window_size=50):
        self._name = name
        self._model = model
        self._history = deque(maxlen=window_size)

    @property
    def history(self):
        return self._history
    
    def detect(self, frame):
        raise NotImplementedError("Subclasses should implement this method")


class MarkerDetector(basicDetector):
    def __init__(self):
        super().__init__(
            name = "ArUco",
            model = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._model)
        centers = None
        if ids is not None:
            centers = np.mean(corners, axis=2).squeeze()
            self._history.append(centers)
        print(f'mean: {self.get_mean_from_history()}', end='\r')
        return str({"centers": centers})

    def is_history_empty(self):
        return len(self._history) == 0

    def get_mean_from_history(self):
        if not self.is_history_empty():
            return np.stack(self._history, axis=0).mean(axis=0)
        else:
            return 0
        

