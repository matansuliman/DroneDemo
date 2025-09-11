import numpy as np


def sym_limits(x): return -x, x

def print_num(x, precision= 2):
    try:
        return f'{x:.{precision}f}'
    except Exception as e:
        print(e)
        return ''

def print_array_of_nums(arr, precision= 2):
    try:
        return ', '.join([print_num(x, precision) for x in arr])
    except Exception as e:
        return ''

def generate_normal_clipped(mean=0, std=1, low=0, high=1, size=1):
    normal = np.random.normal(mean, std, size=size)
    clipped = np.clip(normal, low, high)
    return clipped