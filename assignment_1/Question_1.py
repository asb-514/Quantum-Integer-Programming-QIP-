#!/usr/bin/env python3

from sympy import *
import numpy as np
import timeit

starttime = timeit.default_timer()

x0, x1, x2, x3, x4, x5, x6, x7, z = symbols("x0 x1 x2 x3 x4 x5 x6 x7 z")

eqs = [
    2 * x0
    + x1 * (x1 - 2)
    + 4 * x2 * (x2 - 1)
    + 4 * x3
    + 4 * x4
    + 4 * x5
    + 5 * x6
    + 5 * x4 * x5
    + 2 * x6 * x7
    - z
]

eqs = eqs + [x0 + x3 + x4 + x5 - 2]
eqs = eqs + [x1 + x3 + x6 + x5 - 2]
eqs = eqs + [x2 + x4 + x6 + x7 - 2]
eqs = eqs + [x * (x - 1) for x in [x0, x1, x2, x3, x4, x5, x6, x7]]
result = groebner(eqs, x1, x2, x3, x4, x0, x5, x6, x7, z, order="lex")
result = list(result)

zs = solve(result[-1], z)
zstar = max(zs)
# all possible solutions for z :
print(zs)
# max solution for z satisfing the above equations :
print(zstar)

print(
    "The time taken to execute this script is :",
    timeit.default_timer() - starttime,
)
