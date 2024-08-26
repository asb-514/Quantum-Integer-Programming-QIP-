import numpy as np
import dimod
from typing import Callable
import matplotlib.pyplot as plt

# Objective function definition
def f(x,c):
    return np.dot(c,x)

# Constraints definition
def const(x,A,b):
    return np.array_equiv(np.dot(A,x),b.T) or np.array_equiv(np.dot(A,x),b)


# Define rules to choose augmentation element, either the best one (argmin) or the first one that is found
def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])


def get_feasible(A, b, sampler, samples=20):

    AA = np.dot(A.T, A)
    h = -2.0*np.dot(b.T, A)
    Q = AA + np.diag(h[0])
    offset = np.dot(b.T, b) + 0.0

    # Define Binary Quadratic Model
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(mat=Q, offset=offset)
    # bqm = dimod.BinaryQuadraticModel(Q, offset=offset)

    response = sampler.sample(bqm, num_reads=samples)
    response = response.aggregate()

    filter_idx = [i for i, e in enumerate(response.record.energy) if e == 0.0]

    feas_sols = response.record.sample[filter_idx]

    return feas_sols
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



def postprocess (kernal_sols) :
    """ takes the kernal elements and postprocess them to return the graver elements"""
    graver_sols = []
    flag = [1 for i in range(len(kernal_sols))]
    for i in range(len(kernal_sols)) :
        if flag[i] == 1:
            graver_sols.append(kernal_sols[i])
            for j in range(i,len(kernal_sols)) :
                if is_conformal(kernal_sols[i],kernal_sols[j]):
                    flag[j] = 0
    return graver_sols
# Bisection rules for finding best step size
def bisection(A,b,c,g: np.ndarray, fun: Callable, x: np.ndarray, x_lo: np.ndarray = None, x_up: np.ndarray = None, laststep: np.ndarray = None) -> (float, int):
    if np.array_equal(g, laststep):
        return (fun(x,c), 0)
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
        if fun(x + l*g,c) < fun(x + u*g,c):
            alpha = l
        else:
            alpha = u
        p1 = int(np.floor((l+u)/2) - 1)
        p2 = int(np.floor((l+u)/2))
        p3 = int(np.floor((l+u)/2) + 1)
        if fun(x + p1*g,c) < fun(x + p2*g,c):
            u = int(np.floor((l+u)/2))
        elif fun(x + p3*g,c) < fun(x + p2*g,c):
            l = int(np.floor((l+u)/2) + 1)
        else:
            alpha = p2
            break

    if fun(x + l*g,c) < fun(x + u*g,c) and fun(x + l*g,c) < fun(x + alpha*g,c):
        alpha = l
    elif fun(x + u*g,c) < fun(x + alpha*g,c):
        alpha = u

    return (fun(x + alpha*g,c), alpha)
def augmentation(A,b,c,x, grav,x_lo = None, x_up = None , func = f, OPTION: int = 1, VERBOSE: bool = True, itermax: int = 1000) -> (int, float, np.ndarray):
    # Let's perform the augmentation and return the number of steps and the best solution
    # OPTION = 1 # Best augmentation, select using bisection rule

    dist = 1
    gprev = None
    k = 1
    # print(x_up)
    if VERBOSE:
        print("Initial point:", x)
        print("Objective function:",func(x,c))
    while dist != 0 and k < itermax:
        g1, (obj, dist) = argmin(
            bisection(A = A, b = b, c = c,g=e, fun=func, x=x, laststep=gprev, x_lo=x_lo, x_up=x_up) for e in grav)
        x = x + grav[g1]*dist
        gprev = grav[g1]
        if VERBOSE:
            print("Iteration ", k)
            print(g1, (obj, dist))
            print("Augmentation direction:", gprev)
            print("Distanced moved:", dist)
            print("Step taken:", grav[g1]*dist)
            print("Objective function:", obj)
            print(func(x,c))
            print("Current point:", x)
            print("Are constraints satisfied?", const(x,A,b))
        else:
            if k%50 == 0:
                print(k)
                print(obj)
        k += 1
    return(k,obj,x)

def walk(graver_sols,feas_sols,A,b,c) :
    init_obj = np.zeros((len(feas_sols),1))
    iters_full = np.zeros((len(feas_sols),1))
    final_obj_full = np.zeros((len(feas_sols),1))
    for i,sol in enumerate(feas_sols):
        if not const(sol,A,b):
            print("Infeasible")
            pass
        init_obj[i] = f(sol,c)
        iter, f_obj, xf = augmentation(A = A, b = b, c = c,grav = graver_sols, func = f, x = sol, OPTION=1,VERBOSE=False)
        iters_full[i] = iter
        final_obj_full[i] = f_obj
    return init_obj, final_obj_full ,iters_full
def pplot (init_obj, final_obj_full,iters_full,graver_sols):
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
    print("The max value of objective function is (with ", len(graver_sols)," Graver elements)  : ", (-1*min(final_obj_full)[0]))
    plt.show()
