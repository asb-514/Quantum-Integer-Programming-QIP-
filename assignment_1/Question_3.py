#!/usr/bin/env ipython


from sympy import *
import timeit
import numpy as np


# this script takes around 60-70 seconds
starttime = timeit.default_timer()
left = -1
right = 2

# we create variables x0-1, x00, x01, x02, x1-1, ..... x71,x72. if varibale xji = 1, then it implies that the xj (can be x0, x1,x2,...x7) equal to i is a feasible solution. if x32 = 1 then x3 = 2 is a feasible solution.


# creating varibles
x = [symbols("x{0}{1}".format(j, i)) for j in range(8) for i in range(left, right + 1)]
# print(x)
eqs = []

a = [
    [2, 0, 0, 1, 1, 1, 0, 0],
    [1, 1, 2, 1, 0, -3, 1, 0],
    [0, -1, 1, 3, 1, 0, 1, 1],
    [0, 0, 1, 2, 0, 0, 2, 1],
]

b = [2, 1, 2, 1]

# constraint Ax = B
for j in range(len(a)):
    eq = 0
    for i in range(len(x)):
        eq = eq + a[j][i // (right - left + 1)] * x[i] * (i % (right - left + 1) + left)
    eq = eq - b[j]
    eqs.append(eq)


# constraint that xij*(xij - 1) = 0, each variable can be zero or one.
for i in range(len(x)):
    eqs.append(x[i] * (x[i] - 1))

# if x10 = 1 then x1-1, x11, x12 must be zero, so we add the constraint, x1-1 + x10 + x11 + x12 = 1.
for j in range(8):
    eq = 0
    for i in range(right - left + 1):
        eq = eq + x[i + j * (right - left + 1)]
    eq = eq - 1
    eqs.append(eq)

print(eqs)
result = groebner(eqs, x, order="lex")
result = list(result)
print(result)
resx7 = solve(
    (
        result[-1],
        result[-2],
        result[-3],
        result[-4],
        result[-5],
        result[-6],
        result[-7],
    ),
    (x[-1], x[-2], x[-3], x[-4]),
)
print()
print("possible solutions for (x7-1, x70, x71, x72) : ")
print(resx7)
print(
    "The time taken to execute this script is :",
    timeit.default_timer() - starttime,
)
