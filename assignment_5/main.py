# Import the Dwave packages dimod and neal
import neal
import numpy as np
import helper

# given constraints           transformed constraints
# 0 <= eq1 <= 15               eq1 - y1 - 2*y2 - 4*y3 -8*y4 = 0
# 1 <= eq2 <= 2                eq2 -z1 = 1
A = np.array([[-2,3,9,-8,12,6,5,-1,-4,6,-1,-2,-4,-8,0], [-1,6,-9,10,-4,5,8,-2,-8,4,0,0,0,0,-1]])

b = np.array([[0],[1]])
dummyb = np.array([[0],[0]])

c = np.array([2,5,9,14,12,1,4,1,9,2,0,0,0,0,0])

# maximising np.dot(c,x) is equivalent to minimising np.dot(-1*c , x)
c = [c[i] * -1 for i in range(len(c))]

simAnnSampler = neal.SimulatedAnnealingSampler()
# getting the feasible solutions Ax = b
feas_sols = helper.get_feasible(A, b, sampler=simAnnSampler, samples = 300)
# getting the kernal solutions Ax = 0
kernal_sols = helper.get_feasible(A, dummyb, sampler=simAnnSampler, samples = 300)
# postprocessing the kernal elements to get partial graver basis 
graver_sols = helper.postprocess(kernal_sols)

# walking the grid to reach local minima from all feasible points
init_obj, final_obj_full ,iters_full= helper.walk(graver_sols,feas_sols,A,b,c)

# ploting the initial points and the final points 
helper.pplot(init_obj, final_obj_full,iters_full,graver_sols)


#considering only 2 elements in partial graver basis
graver_sols = graver_sols[ : 2 ]

# walking on the feasible grid to reach local minima but this time only considering 2 graver basis
init_obj, final_obj_full ,iters_full= helper.walk(graver_sols,feas_sols,A,b,c)


# ploting the initial points and the final points 
helper.pplot(init_obj, final_obj_full,iters_full,graver_sols)
# whether we reach the optimal solution considering only 2 graver elements depends on how many starting/feasible solutions are considered
# if we consider high number of feasible solutions then there is a higher chance to reach global extrema, if low number of staring points
# are only taken into consideration then the chances of reaching golbal extrema is low
