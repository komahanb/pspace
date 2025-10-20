import numpy as np


def ensure_scalar_pointer(values, dtype=np.double):
    """Return a contiguous floating array and its data pointer."""
    arr = np.ascontiguousarray(values, dtype=dtype)
    return arr, arr.ctypes.data


def ensure_int_pointer(values, dtype=np.intc):
    """Return a contiguous integer array and its data pointer."""
    arr = np.ascontiguousarray(values, dtype=dtype)
    return arr, arr.ctypes.data
