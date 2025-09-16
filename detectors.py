import cv2
import numpy as np
from collections import deque

from helpers import *

from logger import LOGGER
from config import CONFIG


class BasicDetector:
    def __init__(self, model):
        self._model = model
        self._history = deque(maxlen=CONFIG["Detector"]["history_max_len"])

    @property
    def history(self):
        return list(self._history)

    def clear_history(self, keep=CONFIG["Detector"]["long_term_len"] / 2):
        tail = list(self._history)[-int(keep) :]
        self._history.clear()
        self._history.extend(tail)

    def is_empty(self):
        return len(self._history) == 0

    def is_full(self):
        return len(self._history) == self._history.maxlen

    def get_last(self):
        return self._history[-1] if not self.is_empty() else np.array([np.inf, np.inf])

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def detect(self, frame, curr_height):
        raise NotImplementedError("Subclasses should implement this method")


class ArUcoMarkerDetector(BasicDetector):
    def __init__(self):
        super().__init__(model=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self._tol_stddev = CONFIG["Detector"]["tol_stddev"]
        self._px_to_meter = CONFIG["Detector"]["px_to_meter"]
        self._height_trained = CONFIG["Detector"]["height_trained"]
        self._image_size_trained = CONFIG["Detector"]["image_size_trained"]
        LOGGER.info(f"\t\t\t\tDetector: Initiated {self.__class__.__name__}")

    def get_stddev(self, mode="long-term"):
        if self.is_empty():
            return np.inf
        match mode:
            case "long-term":
                history = np.array(self._history)[
                    -CONFIG["Detector"]["long_term_len"] :
                ]
            case "short-term":
                history = np.array(self._history)[
                    -CONFIG["Detector"]["short_term_len"] :
                ]
            case _:
                raise NotImplementedError
        std_xy = np.stack(history.std(axis=0, ddof=0))
        return np.round(std_xy, CONFIG["Detector"]["round_precision"])

    def is_stable(self, mode="long-term"):
        if self.is_empty():
            return False
        return sum(self.get_stddev(mode)) <= self._tol_stddev

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"\tlast detection: {print_for_gui(self.get_last())}"
        status += f"\tcounter: {len(self._history)}/{self._history.maxlen}\n"
        status += f"\tstddev long term: {print_for_gui(self.get_stddev())}"
        status += f"\t\tis stable long term: {self.is_stable()}\n"
        status += f"\tstddev short term: {print_for_gui(self.get_stddev(mode= 'short-term'))}"
        status += f"\tis stable short term: {self.is_stable(mode= 'short-term')}\n"
        return status

    def detect(self, frame, curr_height):
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

            # calculate coef
            height_ratio = curr_height / self._height_trained
            coef = height_ratio * self._px_to_meter
            centers *= coef

            image_size_trained_x, image_size_trained_y = self._image_size_trained
            coef_x = w / image_size_trained_x
            coef_y = h / image_size_trained_y

            centers[0] /= coef_x
            centers[1] /= coef_y

            if centers.shape != (2,):
                return

            val = np.round(centers, CONFIG["Detector"]["round_precision"])
            self._history.append(val)
