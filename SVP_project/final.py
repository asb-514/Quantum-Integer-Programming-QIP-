from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import itertools
import random
import time


# Define rules to choose augmentation element, either the best one (argmin) or the first one that is found
def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])


def greedy(iterable):
    for i, val in enumerate(iterable):
        if val[1] != 0:
            return i, val
    else:
        return i, val

# Bisection rules for finding best step size

def bisection(
    g: np.ndarray,
    fun: Callable,
    x: np.ndarray,
    x_lo: np.ndarray = None,
    x_up: np.ndarray = None,
    laststep: np.ndarray = None,
) -> (float, int):
    if np.array_equal(g, laststep):
        return (fun(x), 0)
    if x_lo is None:
        x_lo = np.zeros_like(x)
    if x_up is None:
        x_up = np.ones_like(x) * max(x) * 2

    u = max(x_up) - min(x_lo)
    l = -(max(x_up) - min(x_lo))
    for i, gi in enumerate(g):
        if gi >= 1:
            if np.floor((x_up[i] - x[i]) / gi) < u:
                u = int(np.floor((x_up[i] - x[i]) / gi))
            if np.ceil((x_lo[i] - x[i]) / gi) > l:
                l = int(np.ceil((x_lo[i] - x[i]) / gi))
        elif gi <= -1:
            if np.ceil((x_up[i] - x[i]) / gi) > l:
                l = int(np.ceil((x_up[i] - x[i]) / gi))
            if np.floor((x_lo[i] - x[i]) / gi) < u:
                u = int(np.floor((x_lo[i] - x[i]) / gi))
    alpha = u

    while u - l > 1:
        if fun(x + l * g) < fun(x + u * g):
            alpha = l
        else:
            alpha = u
        p1 = int(np.floor((l + u) / 2) - 1)
        p2 = int(np.floor((l + u) / 2))
        p3 = int(np.floor((l + u) / 2) + 1)
        if fun(x + p1 * g) < fun(x + p2 * g):
            u = int(np.floor((l + u) / 2))
        elif fun(x + p3 * g) < fun(x + p2 * g):
            l = int(np.floor((l + u) / 2) + 1)
        else:
            alpha = p2
            break

    if fun(x + l * g) < fun(x + u * g) and fun(x + l * g) < fun(x + alpha * g):
        alpha = l
    elif fun(x + u * g) < fun(x + alpha * g):
        alpha = u

    return (fun(x + alpha * g), alpha)

# We can just have a single step move (works well with greedy approach)
def single_move(
    g: np.ndarray,
    fun: Callable,
    x: np.ndarray,
    x_lo: np.ndarray = None,
    x_up: np.ndarray = None,
    laststep: np.ndarray = None,
) -> (float, int):
    if x_lo is None:
        x_lo = np.zeros_like(x)
    if x_up is None:
        x_up = np.ones_like(x) * max(x) * 2

    alpha = 0

    if (x + g <= x_up).all() and (x + g >= x_lo).all():
        if fun(x + g) < fun(x):
            alpha = 1
    elif (x - g <= x_up).all() and (x - g >= x_lo).all():
        if fun(x - g) < fun(x) and fun(x - g) < fun(x + g):
            alpha = -1

    return (fun(x + alpha * g), alpha)

A = [[64,218,133],[71,205,111],[28, -48, -84]]
A = np.array(A)
print(A.tolist())
# r = graver("mat", A.tolist())
# np.save(r'graver.npy', r)
