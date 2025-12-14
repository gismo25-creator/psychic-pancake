FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

def generate_fibonacci_grid(lower, upper):
    diff = upper - lower
    return [lower + diff * lvl for lvl in FIB_LEVELS]
