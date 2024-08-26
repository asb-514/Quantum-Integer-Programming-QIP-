# the output of this code is the dictionary asked in the assignment question one and 3 section which are :
# first section - this section has the output solution of the quantum annealer
# second section - this section has the output of the quantum annealer + polynomial time processing to reach closer to global maxima/miniaj
# third section - this section has the given solution (global minima) of the data
import sympy
import numpy as np
import helper
from dwave.system import DWaveSampler, EmbeddingComposite


# the data is 0-th input is taken from the assingment and other inputs/testcases are from "https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html"
testcase = 5  # you can enter the number [0,8] correspondingly different input files will be used for data
filename = "input" + str(testcase)

#input file format : first line contains the number of value weight pairs(say = n), second line contains the max weight allowed to be carried
#next n lines contain value weigth pairs in order, next n lines contains the solution to which of the given pair to pick(represented by 1).

# reading input
number_of_object, Max_weight, value_weight, B, solution = helper.read_input(filename)

# Creating variables
symbolNameList, symbolList = helper.Creating_variables(number_of_object, B)

# nessesary equtions
sigma_vi = sum(value_weight[i][0] for i in range(number_of_object))
sigma_wi = sum(value_weight[i][1] for i in range(number_of_object))
sigma_si_wi = sum(
    (symbolList[i] + 1) * (0.5) * value_weight[i][1] for i in range(number_of_object)
)
sigma_si_vi = sum(
    (symbolList[i] + 1) * (0.5) * value_weight[i][0] for i in range(number_of_object)
)
slack_weight = sum(
    (pow(2, i) * (symbolList[number_of_object + i] + 1) * (0.5)) for i in range(B)
)
slack_weight = sum(
    [
        slack_weight,
        (Max_weight - pow(2, B) + 1) * ((symbolList[number_of_object + B] + 1) * (0.5)),
    ]
)

# objecitve function
maximize = sigma_si_vi

# constants
if testcase >= 5 :
    a = 100
    rho = 100000
else :
    a = 10
    rho = 10000

# constraints
constraint = sigma_si_wi + slack_weight - (Max_weight)

# QUBO equation
qubo_eq = sum(
    [
        -1 * a * maximize,
        rho * constraint * constraint,
    ]
)
qubo_eq = sympy.expand(qubo_eq)
result_dict_unreduced = qubo_eq.as_coefficients_dict()

result_dict = helper.Reduce_dictionary(result_dict_unreduced,number_of_object,B)

h, J = helper.fill_h_J(result_dict,number_of_object,B)

print("Assignment Question one")
print()
print(result_dict)
print()
print("Assignment Question two")
print()


num_reads = 1000  # Number of samples you want to obtain

# Create a DWaveSampler
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising(h, J, num_reads=num_reads, return_embedding=True)

best_solution = response.first.sample
txt = "quatum annealer output"
x = txt.center(100, "-")
print(x)

value,weight,slack,selected = helper.Printer(best_solution,h,J,number_of_object,B,Max_weight,value_weight)

txt = "quatum annealer output + Polynomial time Postprocessing"
x = txt.center(100, "-")
print(x)
best_solution = helper.Postprocessing(number_of_object,best_solution,value_weight,slack)

value,weight,slack,selected = helper.Printer(best_solution,h,J,number_of_object,B,Max_weight,value_weight)

txt = "Solution given in the data set"
x = txt.center(100, "-")
print(x)
full_solution = helper.actual_solution(value_weight,number_of_object,B,solution,Max_weight)
value,weight,slack,selected = helper.Printer(full_solution,h,J,number_of_object,B,Max_weight,value_weight)
