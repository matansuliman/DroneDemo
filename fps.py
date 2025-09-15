import time

from helpers import *

from environment import ENVIRONMENT


class BasicFPS:
    def __init__(self, update_rate=1 / ENVIRONMENT.dt):
        self._period = 1 / update_rate
        self._curr_fps = 0
        self._target_fps = 1 / self._period
        self._fps_t0 = time.perf_counter()
        self._fps_cnt = 0

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
        time.sleep(self._period)

    def __str__(self):
        res = ""
        res += f"\t\tcurrent fps: {print_num(self._curr_fps)}"
        res += f"\t\ttarget fps: {print_num(self._target_fps)}\n"
        return res


class AdvanceFPS(BasicFPS):
    def __init__(self, period=ENVIRONMENT.dt):
        super().__init__(period=period)
        self._time = time.perf_counter()
        self._remain = 0

    @property
    def curr_fps(self):
        return self._curr_fps

    @property
    def target_fps(self):
        return self._target_fps

    def reset(self):
        self._time = time.perf_counter()
        self._fps_t0 = time.perf_counter()
        self._fps_cnt = 0
        self._remain = 0

    def _update_curr_fps(self):
        self._fps_cnt += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._curr_fps = self._fps_cnt / (now - self._fps_t0)
            self._fps_cnt, self._fps_t0 = 0, now

    def maintain_update_rate(self):
        self._update_curr_fps()
        # maintain fps
        self._time += self._period
        self._remain = self._time - time.perf_counter()
        if self._remain > 0:
            time.sleep(self._remain)
        time.sleep(self._period)

    def __str__(self):
        res = ""
        res += f"\t\tcurrent fps: {print_num(self._curr_fps)}"
        res += f"\ttarget fps: {print_num(self._target_fps)}"
        if self._remain > 0:
            res += "\tsleep"
        res += f"\tremain: {self._remain}\n"
        return res
