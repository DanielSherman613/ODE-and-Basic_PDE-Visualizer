import numpy as np


def ripple(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.sin(r)


def saddle(x, y):
    return 0.1 * (x**2 - y**2)


def sinc2d(x, y):
    r = np.sqrt(x**2 + y**2) + 1e-9
    return np.sin(r) / r


SURFACE_PRESETS = {
    "ripple": ripple,
    "saddle": saddle,
    "sinc2d": sinc2d,
}