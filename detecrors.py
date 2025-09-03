import time
import cv2
import numpy as np
from PySide6.QtCore import QObject

from logger import LOGGER
from config import CONFIG


class BasicDetector:
    def __init__(self, model):
        self._model = model

    def detect(self, frame):
        raise NotImplementedError("Subclasses should implement this method")


class ArUcoMarkerDetector(BasicDetector):
    def __init__(self):
        super().__init__(model=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        LOGGER.info(f"\t\t\t\tDetector: Initiated {self.__class__.__name__}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._model)

        if ids is not None:
            # calculate centers from corners
            centers = np.mean(corners, axis=2).squeeze()

            # convert from top left to center
            h, w = frame.shape[:2]
            img_center = (w / 2.0, h / 2.0)
            centers[0] -= img_center[0]
            centers[1] -= img_center[1]

            # flip y axis
            centers[1] = -centers[1]

            # convert from px to meters
            centers *= CONFIG["Detector"]["px_to_meter"]

            # return detection with time stamp
            return time.time(), np.round(centers, 2)

        else:
            return None
