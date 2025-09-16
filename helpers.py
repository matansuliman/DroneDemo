import numpy as np

from logger import LOGGER
from config import CONFIG


def sym_limits(x):
    return -x, x


def _print_num(x: float | int, precision):
    return f"{x:.{precision}f}"

def _print_array_of_nums(arr: list, precision):
    return "  ".join([_print_num(x, precision) for x in arr])

def print_for_gui(data, precision= CONFIG["gui"]["precision"]):
    t = type(data)
    if isinstance(data, list) or isinstance(data, tuple) or isinstance(data, np.ndarray):
        return _print_array_of_nums(data, precision)
    elif isinstance(data, int) or isinstance(data, float):
        return _print_num(data, precision)
    else:
        print(t)
        return "not implemented"


def generate_normal_clipped(mean=0, std=1, low=0, high=1, size=1):
    normal = np.random.normal(mean, std, size=size)
    clipped = np.clip(normal, low, high)
    return clipped
