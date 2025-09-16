import numpy as np
from collections import deque

from helpers import *

from detectors import ArUcoMarkerDetector

from logger import LOGGER


class BasicPredictor:
    def __init__(self, model):
        LOGGER.info("\t\t\tPredictor: Initiating")
        self._model = model()
        self._history = deque([[0, 0, 0]])
        self._prediction = np.array([0, 0, 0], dtype=float)

    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return list(self._history)

    def is_empty(self):
        return len(self._history) == 0

    def get_last(self):
        return self._history[-1] if not self.is_empty() else None

    def predicted(self):
        return self.get_last() is not None

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    def status(self):
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self):
        raise NotImplementedError("Subclasses should implement this method")


class ArUcoMarkerPredictor(BasicPredictor):
    def __init__(self):
        super().__init__(model=ArUcoMarkerDetector)
        LOGGER.info(f"\t\t\tPredictor: Initiated {self.__class__.__name__}")

    def is_model_stable(self, mode="long-term"):
        return self._model.is_stable(mode=mode)

    def stream_to_model(self, frame, curr_height):
        self.model.detect(frame, curr_height)

    def status(self):
        status = f"{self.__class__.__name__} status:\n"
        status += f"model: {self._model.status()}"
        status += f"\tlast prediction: {print_for_gui(self.get_last())}"
        status += f"\taccumulated prediction: {print_for_gui(self._prediction)}\n"
        return status

    def get_last_from_model(self):
        return self._model.get_last()

    def predict(self):
        if self._model.is_full() and self.is_model_stable():
            LOGGER.debug("Predictor: model is full and stable")
            LOGGER.debug("Predictor: predicting")

            # calculate mean of model history and add dim(z)
            pred = np.append(np.mean(self._model.history, axis=0), 0)

            self._prediction += pred
            self._history.append(self._prediction)
            LOGGER.debug(f"Predictor: prediction = {self._prediction}")
            self._model.clear_history()
