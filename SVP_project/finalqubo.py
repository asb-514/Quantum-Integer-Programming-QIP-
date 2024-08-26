import sympy
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import neal

file_path = 'experimental_sieve/fplll/inp'
with open(file_path, 'r') as file:
    # Read lines from the file
    lines = file.readlines()

# Parse each line as a list of integers, handling brackets and commas
list_of_lists = []
for line in lines:
    # Remove brackets and split based on commas and spaces
    elements = line.replace('[', '').replace(']', '').replace(',', ' ').split()
    
    # Convert each element to an integer
    int_elements = [int(element) for element in elements]
    
    # Append the list of integers to the result
    list_of_lists.append(int_elements)

def Creating_variables(number_of_object, B):
    symbolNameList = []
    symbolList = []
    for i in range(number_of_object):
        tsymbolNameList = []
        tsymbolList = []
        for j in range(B + 1) :
            name = "X_" + str(i) + "_"+ str(j)
            tsymbolNameList.append(name)
            tsymbolList.append(sympy.Symbol(name))
        symbolNameList.append(tsymbolNameList)
        symbolList.append(tsymbolList)
        

    return (symbolNameList, symbolList)

def xieq(x, symbolList, maxa):
    """TODO: Docstring for xieq.

    :x: TODO
    :returns: TODO

    """
    ans = 0
    ans = sum ([(symbolList[x][i] + 1) * (0.5) * pow(2,i) for i in range(len(symbolList[x]) - 1)])
    ans = sum([ans, (symbolList[x][len(symbolList[x]) - 1] + 1) * (0.5) * (2 * maxa - pow(2,len(symbolList[x]) - 1) + 1)])
    ans = sum([ans,-1*maxa])
    return ans


def separate_numbers(input_string):
    # Split the input string by underscores
    parts = input_string.split('_')

    # Filter out non-numeric parts and convert them to integers
    numbers = [int(part) for part in parts if part.isdigit()]

    return numbers






# Display the result
list_of_lists = np.matrix(list_of_lists)
A = np.array(list_of_lists.T).tolist()
maxa = 0
floorlog = 0;
po = 1;
for i in range(len(A)):
    for j in range(len(A[0])):
        maxa = max(maxa, abs(A[i][j]))
# print(maxa)  # range = [-maxa, maxa]
while po <=  maxa : 
    floorlog += 1
    po *= 2
symbolNameList, symbolList = Creating_variables(len(A[0]), floorlog)
print(floorlog, len(symbolList[0]))

b = [ sum([A[i][j]*xieq(j,symbolList,maxa) for j in range(len(A[0]))]) for i in range(len(A)) ]
minimise = sum([b[i]*b[i] for i in range(len(b))])
# minimise = 1e9*(1-minimise) + minimise*minimise
qubo_eq = sympy.expand(minimise)
result_dict_unreduced = qubo_eq.as_coefficients_dict()
print(result_dict_unreduced)
print(floorlog)
result_dict = {
    (i, j): 0
    for i in range(len(symbolList)*(floorlog + 1))
    for j in range(i,len(symbolList)*(floorlog + 1))
}
for key in result_dict_unreduced.keys():
    if type(key) == sympy.core.power.Pow:
        # s_i**2 terms so always s_i**2 = 1 so it contributes to constant term
        pass
    elif type(key) == sympy.core.symbol.Symbol:
        # s_i terms
        result_dict[int(str(key)[2:]), int(str(key)[2:])] += result_dict_unreduced[
            key
        ]
    elif type(key) == sympy.core.mul.Mul:
        # cross terms
        one = separate_numbers(str(key.args[0]))
        two = separate_numbers(str(key.args[1]))
        ind2 = max(one[0]*(floorlog + 1) + one[1], two[0]*(floorlog + 1) + two[1])
        ind1 = min(one[0]*(floorlog + 1) + one[1], two[0]*(floorlog + 1) + two[1])
        result_dict[ind1, ind2] += result_dict_unreduced[key]
    elif type(key) == sympy.core.numbers.One:
        # constant
        pass
    else:
        print("ERROR : Possibily not QUBO")
print(result_dict)
h = {}  # Dictionary for linear terms (h_i)
J = {}  # Dictionary for quadratic terms (J_ij)
# Set h_i values
for i in range(len(symbolList)*(floorlog + 1)):  # Assuming n is the number of variables
    h[i] = result_dict[(i, i)]

# Set J_ij values
for (i, j), value in result_dict.items():
    if i != j:
        J[(i, j)] = value



num_reads = 1000  # Number of samples you want to obtain

# Create a DWaveSampler
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising(h, J, num_reads=num_reads, return_embedding=True)

best_solution = response.first.sample
print(best_solution)
simpeq = (qubo_eq)
btemp = (b)
for u in best_solution :
    simpeq = simpeq.subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_solution[u])
    minimise = minimise.subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_solution[u])
    b = [b[i].subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_solution[u]) for i in range(len(b))]
print(simpeq)
#print(qubo_eq)
print(minimise)
print(b)

