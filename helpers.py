def sym_limits(x): return -x, x

def print_array_of_nums(arr):
    try:
        return '  '.join([f'{x:.4f}' for x in arr])
    except Exception as e:
        return ''