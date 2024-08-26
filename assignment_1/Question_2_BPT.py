#!/usr/bin/env ipython

# solving the second question using BPT method, this code take 15-20 seconds to execute
from sympy import *
import numpy as np
import timeit

starttime = timeit.default_timer()
x0, x1, x2, x3, x4, x5, x6, x7 = symbols("x0 x1 x2 x3 x4 x5 x6 x7")

eqs = []
eqs = eqs + [2 * x0 + x3 + x4 + x5 - 2]
eqs = eqs + [x1 + x0 + 2 * x2 + x3 + -3 * x5 + x6 - 1]
eqs = eqs + [-1 * x1 + x2 + 3 * x3 + x4 + x6 + x7 - 2]
eqs = eqs + [x2 + 2 * x3 + 2 * x6 + x7 - 1]

left = -1
right = 2
# assuming all the solutions for x0,x1 ... x7 lie in range [left, right], in this case all Xi belongs to [-1,2].

# instead of adding x*(x-1) to constraints, now we add (x-left)*(x-(left + 1)) ... *(x - right).
for x in [x0, x1, x2, x3, x4, x5, x6, x7]:
    eq = 1
    for i in range(left, right + 1):
        eq *= x - i
    eqs.append(eq)
# print(eqs)
result = groebner(eqs, x0, x1, x2, x3, x4, x5, x6, x7, order="lex")
result = list(result)

# print(result)
zs = solve(result[-1], x7)
# possible solutions for x7 in range [-1,2]:
print(zs)


# below shows the process to backpropagate
# print()
# zsx6 = solve(result[-2].subs({(x7, zs[0])}), x6)
# print(solve(result[-3].subs({(x7, zs[0]), (x6, zsx6[0])})))
print(
    "The time taken to execute this script is :",
    timeit.default_timer() - starttime,
)
