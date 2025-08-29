import time
import cv2
import numpy as np
from collections import deque
from PySide6.QtCore import Signal, QObject

class BasicDetector:
    def __init__(self, info = '', model = None, k_sec=15):
        self._info = info
        self._model = model
        self._history = deque(maxlen=k_sec*10)
        self._k_sec = k_sec
        self._predicted = True

    @property
    def history(self):
        return list(self._history)

    @property
    def predicted(self):
        return self._predicted

    @predicted.setter
    def predicted(self, predicted):
        self._predicted = predicted

    def is_valid(self):
        return self._history.maxlen == len(self._history)

    def get_last(self):
        return self._history[-1][1]
    
    def get_mean_from_history(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_std_from_history(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def detect(self, frame):
        raise NotImplementedError("Subclasses should implement this method")
    
    

class MarkerDetector(BasicDetector):
    def __init__(self):
        super().__init__(
            info = "ArUco",
            model = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )

    def _filtered_centers(self):
        now = time.time()
        cutoff = now - self._k_sec
        return [c for (t, c) in self._history if t >= cutoff]

    def get_mean_from_history(self):
        centers = self._filtered_centers()
        mean_xy = np.stack(centers, axis=0).mean(axis=0)
        return np.array([round(float(mean_xy[0]),2), round(float(mean_xy[1]),2)])

    def get_std_from_history(self):
        centers = self._filtered_centers()
        std_xy = np.stack(centers, axis=0).std(axis=0, ddof=0)
        return np.array([round(float(std_xy[0]),2), round(float(std_xy[1]),2)])

    def detect(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._model)
        centers = None
        res_str = ''

        if ids is not None:
            centers = np.mean(corners, axis=2).squeeze()
            # center px
            centers[0] -= 159.5
            centers[1] -= 119.5

            # swap x,y
            temp = centers[0]
            centers[0] = centers[1]
            centers[1] = temp

            # append to history with timestamp
            self._history.append((time.time(), centers))
            res_str = f'{self._info} detected!'

        if self.is_valid():
            res_str += f'\nstd over {self._k_sec}sec is {self.get_std_from_history()}'
            if self.is_stable(): res_str += f'\nstable!'
        return res_str

    def is_stable(self):
        std_k_sec  = self.get_std_from_history()
        return sum(std_k_sec) <= 2
        