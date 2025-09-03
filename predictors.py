import time
import cv2
import numpy as np
from collections import deque
from PySide6.QtCore import QObject

import logging
logger = logging.getLogger("app")


class BasicPredictor:
    def __init__(self, info = '', model = None):
        logger.info("\t\t\tPredictor: Initiating")
        self._info = info
        self._model = model()
        self._prediction = None

    @property
    def model(self):
        return self._model

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    @property
    def predicted(self):
        return self._prediction is not None

    def predict(self, frame):
        raise NotImplementedError("Subclasses should implement this method")


class ArUcoMarkerPredictor(BasicPredictor):
    def __init__(self, model= None, time_frame: int= 15):
        super().__init__(info= "ArUco Marker Predictor", model= model)
        self._time_frame = time_frame
        self._history = deque(maxlen= 10 * time_frame)
        self._tol_std = 0.2
        logger.info(f"\t\t\tPredictor: Initiated {self.__class__.__name__}")

    @property
    def history(self):
        return list(self._history)

    def _is_full(self):
        return self._history.maxlen == len(self._history)

    def get_last(self):
        return self._history[-1][1]

    def _get_std(self):
        # filter by time_frame
        now = time.time()
        cutoff = now - self._time_frame
        centers = [c for (t, c) in self._history if t >= cutoff]

        std_xy = np.stack(centers, axis=0).std(axis=0, ddof=0)
        return np.array([round(float(std_xy[0]),2), round(float(std_xy[1]),2)])

    def _is_stable(self):
        if sum(self._get_std()) <= self._tol_std:
            logger.debug(f"Predictor: findings are stable")
            return True
        else:
            return False

    def predict(self, frame):
        # always update detections
        detection = self.model.detect(frame)
        if detection is not None:
            self._history.append(detection)

        # predict only once
        if self._prediction is not None:
            return f"prediction is {self._prediction}"

        # if not predicted yet and can predict
        elif self._is_full() and self._is_stable():
            logger.debug("Predictor: full and stable")

            # filter by time_frame
            now = time.time()
            cutoff = now - self._time_frame
            centers = [c for (t, c) in self._history if t >= cutoff]

            # calculate mean of detection history
            self._prediction = np.append(np.round(np.mean(centers, axis=0), 2), 0)
            logger.debug(f"Predictor: prediction = {self._prediction}")
            return f"prediction is {self._prediction}"

        # if cant predict, send detection if exist
        elif detection is not None:
            return (f"detection is {detection[1]} \n"
                    f"std is {list(self._get_std()), np.round(sum(self._get_std()), 2)} \n"
                    f"history len {len(self._history)}")

        else:
            return ""