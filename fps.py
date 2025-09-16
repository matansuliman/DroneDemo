import time
from loop_rate_limiters import RateLimiter, logging

from helpers import *

from environment import ENVIRONMENT


class BasicFPS:
    def __init__(self, frequency= (1 / ENVIRONMENT.dt)):
        self._period = 1 / frequency

        self._curr_fps = 0
        self._target_fps = frequency

        self._fps_t0 = time.perf_counter()
        self._fps_cnt = 0

        self.rate_limiter = RateLimiter(frequency= frequency)
        logging.disable_warnings()

    @property
    def period(self):
        return self._period

    @property
    def curr_fps(self):
        return self._curr_fps

    @property
    def target_fps(self):
        return self._target_fps

    def _update(self):
        self._fps_cnt += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._curr_fps = self._fps_cnt / (now - self._fps_t0)
            self._fps_cnt, self._fps_t0 = 0, now

    def maintain(self):
        self._update()
        #time.sleep(self._period)
        self.rate_limiter.sleep()

    def __str__(self):
        res = ""
        res += f"\t\tcurrent fps: {print_for_gui(self._curr_fps)}"
        res += f"\t\ttarget fps: {print_for_gui(self._target_fps)}\n"
        return res