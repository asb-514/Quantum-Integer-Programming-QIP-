from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import itertools
import random


def to_lawrence_polynomial(coef, xs, ys):
    """
    Function to define a single column of the coefficient as a polynomial
    """
    res1 = 1
    res2 = 1
    for i in range(len(coef)):
        if coef[i] >= 0:
            res1 = res1 * xs[i] ** coef[i]
            res2 = res2 * ys[i] ** coef[i]
        else:
            res2 = res2 * xs[i] ** (-coef[i])
            res1 = res1 * ys[i] ** (-coef[i])
    res = res1 - res2
    return res


def lawrence_polynomial_ideal(A):
    """
    Function to define a the polynomial ideal of the Lawrence lifting of matrix A
    """
    # Find nullspace (kernel) of A
    ker = A.nullspace()

    # Normalize elements of kernel to be integers
    for i in range(len(ker)):
        rationalvector = True
        while rationalvector:
            factor = 1
            for j in ker[i]:
                if j % 1 != 0:
                    factor = min(factor, j % 1)
            if factor == 1:
                rationalvector = False
            else:
                ker[i] = ker[i] / factor

    # Define symbolic variables zs for each row (index 0 in Python)
    sym_str_x = "x:" + str(A.shape[1])
    xs = symbols(sym_str_x)

    # Define symbolic variables ws for each column (index 0 in Python)
    sym_str_y = "y:" + str(A.shape[1])
    ys = symbols(sym_str_y)

    sys = []
    for k in ker:
        sys.append(to_lawrence_polynomial(k, xs, ys))

    return (sys, xs, ys)


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


# change Problem variable to "a","b","c","d" to solve respective problems
Problem = "d"


# Objective function definition
def f(x):
    if Problem == "a":
        return np.dot(c, x)
    elif Problem == "b":
        return np.exp(np.sum(np.dot(c, x)))
    elif Problem == "c":
        ans = 0
        for i in range(len(c)):
            ans += np.log(c[i] + x[i])
        return ans
    elif Problem == "d":
        ans = 0
        for i in range(len(c)):
            ans += np.exp(c[i] * x[i] * x[i])
        return ans


def const(x):
    return np.array_equiv(np.dot(A, x), b.T) or np.array_equiv(np.dot(A, x), b)


A = np.array(
    [
        [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    ]
)

b = np.array([[1], [1], [1]])
c = np.array([2, 4, 4, 4, 4, 4, 5, 4, 5, 6, 5])
x0 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
x_lo = np.zeros_like(x0)
x_up = np.ones_like(x0)

_A = Matrix(A)
_la, _x, _y = lawrence_polynomial_ideal(_A)
# r = [[2, -3, 0, 1], [1, -2, 1, 0], [1, -1, -1, 1], [0, 1, -2, 1], [1, 0, -3, 2]]
# r = np.array(r)
# print(r)
print("Lawrence polynomial ideal")
LA = _la
xs = _x
ys = _y
print(LA)
randvars = tuple(random.sample(xs + ys, len(xs + ys)))
result_lawrence = groebner(LA, randvars, order="lex")
grav = []
for g in result_lawrence:
    for y in ys:
        g = g.subs({(y, 1)})
    grav.append(g)
print("Reduced Graver (Groebner since we computed it with the Lawrence ideal) basis")
_grav = list(groebner(grav, xs, order="grevlex"))

print(_grav)


def convert(eq, xs):
    """
    function to read the powers from the graver basis to get the test set
    """
    eq = eq.as_ordered_terms()
    exp1 = [eq[0].count(var) for var in xs]
    exp2 = [eq[1].count(var) for var in xs]
    res = []
    for i in range(len(exp1)):
        if str(eq)[1] != "-":
            res.append(exp1[i] - exp2[i])
        else:
            res.append(exp2[i] - exp1[i])
    return res


r = []
for g in _grav:
    r.append(convert(g, xs))

r = np.array(r)
print("Test set is :")
print(r)
# Let's perform the augmentation
OPTION = 1  # Best augmentation, select using bisection rule
# OPTION = 2 # Greedy augmentation, select using bisection rule
# OPTION = 3  # Greedy augmentation, select using first found

dist = 1
gprev = None
k = 1
print("Initial point:", x0)
while dist != 0:
    if OPTION == 1:
        g1, (obj, dist) = argmin(
            bisection(e, f, x0, laststep=gprev, x_lo=x_lo, x_up=x_up) for e in r
        )
    elif OPTION == 2:
        g1, (obj, dist) = greedy(
            bisection(e, f, x0, laststep=gprev, x_lo=x_lo, x_up=x_up) for e in r
        )
    elif OPTION == 3:
        g1, (obj, dist) = greedy(single_move(e, f, x0, x_lo=x_lo, x_up=x_up) for e in r)
    else:
        print("Option not implemented")
        break
    x0 = x0 + r[g1] * dist
    gprev = r[g1]
    print("Iteration ", k)
    print(g1, (obj, dist))
    print("Augmentation direction:", gprev)
    print("Distanced moved:", dist)
    print("Step taken:", r[g1] * dist)
    print("Objective function:", obj)
    print("Current point:", x0)
    print("Are constraints satisfied?", const(x0))
    k += 1
