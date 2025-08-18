import time
import cv2
import numpy as np
from collections import deque

class basicDetector():
    def __init__(self,name = "", model = None, k_sec=15):
        self._name = name
        self._model = model
        self._history = deque(maxlen=k_sec*10)
        self._k_sec = k_sec

    @property
    def history(self):
        return list(self._history)
    
    def get_mean_from_history(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_std_from_history(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def detect(self, frame):
        raise NotImplementedError("Subclasses should implement this method")
    
    

class MarkerDetector(basicDetector):
    def __init__(self):
        super().__init__(
            name = "ArUco",
            model = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )
        self._final_adjust = np.array([0,0,0])

    @property
    def final_adjust(self):
        return self._final_adjust
    
    @final_adjust.setter
    def final_adjust(self, final_adjust):
        self._final_adjust = final_adjust

    def _filtered_centers(self):
        now = time.time()
        cutoff = now - self._k_sec
        return [c for (t, c) in self._history if t >= cutoff]

    def get_mean_from_history(self):
        centers = self._filtered_centers()
        if not centers:
            return None
        mean_xy = np.stack(centers, axis=0).mean(axis=0)
        return np.array([round(float(mean_xy[0]),2), round(float(mean_xy[1]),2)])

    def get_std_from_history(self):
        centers = self._filtered_centers()
        if not centers:
            return None
        std_xy = np.stack(centers, axis=0).std(axis=0, ddof=0)
        return np.array([round(float(std_xy[0]),2), round(float(std_xy[1]),2)])

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._model)
        centers = None

        if ids is not None:
            centers = np.mean(corners, axis=2).squeeze()
            centers[0] -= 159.5
            centers[1] -= 119.5
            temp = centers[0]
            centers[0] = centers[1]
            centers[1] = temp
            self._history.append((time.time(), centers))

        mean_k_sec = self.get_mean_from_history()
        std_k_sec  = self.get_std_from_history()
        
        if mean_k_sec is not None:
            return f'std ({self._k_sec}s) = {std_k_sec}, stable for {self._k_sec}s: {str(self.is_stable())}'
        else: 
            return ""
    

    def is_stable(self):
        std_k_sec  = self.get_std_from_history()
        if std_k_sec is None:
            return False
        return sum(std_k_sec) <= 2
        