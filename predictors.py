

from collections import deque
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal



class RelPosPredictor(QObject):
    
    prediction_vector_ready = pyqtSignal(str)

    def __init__(self, window_size=50):
        self._last = None
        self._history = deque(maxlen=window_size)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def on_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if ids is None or len(ids) == 0:
             self._last = None
             self.prediction_vector_ready.emit("No markers detected.")

        else:
            parts = []
            for marker_id, corner in zip(ids.flatten(), corners):
                # compute center pixel
                pts = corner.reshape(-1, 2)
                cx, cy = pts.mean(axis=0)

                # center
                dx, dy = cx-159.5, cy-119.5
                
                
                self._last = (dx, dy, 0)
                self._history.append(self._last)

            # join into one string
            prediction_mean = np.mean(self._history, axis=0)
            dx, dy, dz = prediction_mean
            parts.append(f"Marker @ ({dx:.1f}, {dy:.1f})")
            self.prediction_vector_ready.emit("; ".join(parts))


    def get(self):
        return np.mean(self._history, axis=0)
        

