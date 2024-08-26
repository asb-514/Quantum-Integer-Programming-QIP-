# Import the Dwave packages dimod and neal
import dimod
import neal
import helper
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
import itertools
import random
import time

A = np.array([[-2,3,9,-8,12,6,5,-1,-4,6,-1,-2,-4,-8,0], [-1,6,-9,10,-4,5,8,-2,-8,4,0,0,0,0,-1]])
b = np.array([[0],[1]])
dummyb = np.array([[0],[0]])
c = np.array([2,5,9,14,12,1,4,1,9,2,0,0,0,0,0])
c = [c[i] * -1 for i in range(len(c))]
x0 = np.array([3,0,6,1])
x_lo = np.zeros_like(x0)
x_up = np.ones_like(x0)


simAnnSampler = neal.SimulatedAnnealingSampler()
feas_sols = helper.get_feasible(A, b, sampler=simAnnSampler, samples = 300)
kernal_sols = helper.get_feasible(A, dummyb, sampler=simAnnSampler, samples = 300)
#print(len(feas_sols), ' feasible solutions found.')
#print(len(kernal_sols), ' feasible solutions found.')
# print((feas_sols), ' feasible solutions found.')
# print((kernal_sols), ' feasible solutions found.')
def is_conformal(x, y):
    """
    Check if vector x is conformal to vector y.

    Parameters:
    - x, y: Lists or NumPy arrays representing vectors.

    Returns:
    - True if x is conformal to y, False otherwise.
    """
    # Determine the common length of vectors
    common_length = min(len(x), len(y))

    # Check conformality condition for each common component
    for i in range(common_length):
        xi, yi = x[i], y[i]
        if xi * yi <= 0 or abs(xi) > abs(yi):
            return False

    # If all common components satisfy the conformality condition, return True
    return True
graver_sols = []
flag = [1 for i in range(len(kernal_sols))]
for i in range(len(kernal_sols)) :
    if flag[i] == 1:
        graver_sols.append(kernal_sols[i])
        for j in range(i,len(kernal_sols)) :
            if is_conformal(kernal_sols[i],kernal_sols[j]):
                flag[j] = 0
# print(len(graver_sols))
# print(len(kernal_sols))


# Bisection rules for finding best step size
def bisection(g: np.ndarray, fun: Callable, x: np.ndarray, x_lo: np.ndarray = None, x_up: np.ndarray = None, laststep: np.ndarray = None) -> (float, int):
    if np.array_equal(g, laststep):
        return (fun(x), 0)
    if x_lo is None:
        x_lo = np.zeros_like(x)
    if x_up is None:
        x_up = np.ones_like(x)

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
        if fun(x + l*g) < fun(x + u*g):
            alpha = l
        else:
            alpha = u
        p1 = int(np.floor((l+u)/2) - 1)
        p2 = int(np.floor((l+u)/2))
        p3 = int(np.floor((l+u)/2) + 1)
        if fun(x + p1*g) < fun(x + p2*g):
            u = int(np.floor((l+u)/2))
        elif fun(x + p3*g) < fun(x + p2*g):
            l = int(np.floor((l+u)/2) + 1)
        else:
            alpha = p2
            break

    if fun(x + l*g) < fun(x + u*g) and fun(x + l*g) < fun(x + alpha*g):
        alpha = l
    elif fun(x + u*g) < fun(x + alpha*g):
        alpha = u

    return (fun(x + alpha*g), alpha)
# We can just have a single step move (works well with greedy approach)
def single_move(g: np.ndarray, fun: Callable, x: np.ndarray, x_lo: np.ndarray = None, x_up: np.ndarray = None, laststep: np.ndarray = None) -> (float, int):
    if x_lo is None:
        x_lo = np.zeros_like(x)
    if x_up is None:
        x_up = np.ones_like(x)*max(x)*2

    alpha = 0

    if (x + g <= x_up).all() and (x + g >= x_lo).all():
        if fun(x + g) < fun(x):
            alpha = 1
    elif (x - g <= x_up).all() and (x - g >= x_lo).all():
        if fun(x - g) < fun(x) and fun(x - g) < fun(x + g):
            alpha = -1

    return (fun(x + alpha*g), alpha)
def greedy(iterable):
    for i, val in enumerate(iterable):
        if val[1] != 0:
            return i, val
    else:
        return i, val

# Objective function definition
def f(x):
    return np.dot(c,x)

# Constraints definition
def const(x):
    return np.array_equiv(np.dot(A,x),b.T) or np.array_equiv(np.dot(A,x),b)
# Define rules to choose augmentation element, either the best one (argmin) or the first one that is found
def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])
def augmentation(x, x_lo = None, x_up = None,grav = graver_sols, func = f, OPTION: int = 1, VERBOSE: bool = True, itermax: int = 1000) -> (int, float, np.ndarray):
    # Let's perform the augmentation and return the number of steps and the best solution
    # OPTION = 1 # Best augmentation, select using bisection rule
    # OPTION = 2 # Greedy augmentation, select using bisection rule
    # OPTION = 3 # Greedy augmentation, select using first found

    dist = 1
    gprev = None
    k = 1
    # print(x_up)
    if VERBOSE:
        print("Initial point:", x)
        print("Objective function:",func(x))
    while dist != 0 and k < itermax:
        if OPTION == 1:
            g1, (obj, dist) = argmin(
                bisection(g=e, fun=func, x=x, laststep=gprev, x_lo=x_lo, x_up=x_up) for e in grav)
        elif OPTION == 2:
            g1, (obj, dist) = greedy(
                bisection(g=e, fun=func, x=x, laststep=gprev, x_lo=x_lo, x_up=x_up) for e in grav)
        elif OPTION == 3:
            g1, (obj, dist) = greedy(
                single_move(g=e, fun=func, x=x, x_lo=x_lo, x_up=x_up) for e in grav)
        else:
            print("Option not implemented")
            break
        x = x + grav[g1]*dist
        gprev = grav[g1]
        if VERBOSE:
            print("Iteration ", k)
            print(g1, (obj, dist))
            print("Augmentation direction:", gprev)
            print("Distanced moved:", dist)
            print("Step taken:", grav[g1]*dist)
            print("Objective function:", obj)
            print(func(x))
            print("Current point:", x)
            print("Are constraints satisfied?", const(x))
        else:
            if k%50 == 0:
                print(k)
                print(obj)
        k += 1
    return(k,obj,x)
init_obj = np.zeros((len(feas_sols),1))
iters_full = np.zeros((len(feas_sols),1))
final_obj_full = np.zeros((len(feas_sols),1))
times_full = np.zeros((len(feas_sols),1))
for i,sol in enumerate(feas_sols):
    if not const(sol):
        print("Infeasible")
        pass
    init_obj[i] = f(sol)
    start = time.process_time()
    iter, f_obj, xf = augmentation(grav = graver_sols, func = f, x = sol, OPTION=1,VERBOSE=False)
    times_full[i] = time.process_time() - start
    iters_full[i] = iter
    final_obj_full[i] = f_obj

# print(final_obj_full)
# print(min(final_obj_full))
print("The max value of objective function is (with ", len(graver_sols)," Graver elements)  : ", (-1*min(final_obj_full)[0]))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(init_obj, marker='o', ls='-', label='Initial')
ax1.plot(final_obj_full, marker='*', ls='-', label='Final')
ax1.set_ylabel('Objective function')
ax1.set_xlabel('Feasible solution')


color = 'tab:green'
ax2.set_ylabel('iterations', color=color)
ax2.plot(iters_full, color=color, marker='s', ls='')
ax2.tick_params(axis='y', labelcolor=color)
ax1.legend()
plt.title(f"{len(graver_sols)} Graver elements")
plt.show()
graver_sols = graver_sols[:2]
init_obj = np.zeros((len(feas_sols),1))
iters_full = np.zeros((len(feas_sols),1))
final_obj_full = np.zeros((len(feas_sols),1))
times_full = np.zeros((len(feas_sols),1))
for i,sol in enumerate(feas_sols):
    if not const(sol):
        print("Infeasible")
        pass
    init_obj[i] = f(sol)
    start = time.process_time()
    iter, f_obj, xf = augmentation(grav = graver_sols, func = f, x = sol, OPTION=1,VERBOSE=False)
    times_full[i] = time.process_time() - start
    iters_full[i] = iter
    final_obj_full[i] = f_obj

# print(final_obj_full)
# print(min(final_obj_full))
print("The max value of objective function is (with ", len(graver_sols)," Graver elements)  : ", (-1*min(final_obj_full)[0]))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(init_obj, marker='o', ls='-', label='Initial')
ax1.plot(final_obj_full, marker='*', ls='-', label='Final')
ax1.set_ylabel('Objective function')
ax1.set_xlabel('Feasible solution')


color = 'tab:green'
ax2.set_ylabel('iterations', color=color)
ax2.plot(iters_full, color=color, marker='s', ls='')
ax2.tick_params(axis='y', labelcolor=color)
ax1.legend()
plt.title(f"{len(graver_sols)} Graver elements")
plt.show()
