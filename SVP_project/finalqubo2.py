
from dwave.system import (DWaveSampler, EmbeddingComposite,FixedEmbeddingComposite)
import sympy
import numpy as np
import dimod
import neal
from itertools import product
import dwave_networkx as dnx
from pprint import pprint
from scipy.special import gamma
import math
from collections import Counter
import pandas as pd
from itertools import chain

# Import Matplotlib to generate plots
import matplotlib.pyplot as plt

def plot_enumerate(results, title=None):

    plt.figure()

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by=None)]

    if results.vartype == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(
            ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        plt.xlabel('bitstring for solution')
    else:
        samples = np.arange(len(energies))
        plt.xlabel('solution')

    plt.bar(samples,energies)
    plt.xticks(rotation=90)
    plt.ylabel('Energy')
    plt.title(str(title))
    print("minimum energy:", min(energies))


def plot_energies(results, title=None):
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None)

    plt.xlabel('Energy')
    plt.ylabel('Probabilities')
    plt.title(str(title))
    plt.show()
    print("minimum energy:", min(energies))


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

# print(pow(np.abs(np.linalg.det(list_of_lists)),1/len(list_of_lists)) * len(list_of_lists))
def Creating_variables(number_of_object, B):
    symbolNameList = []
    symbolList = []
    for i in range(number_of_object):
        tsymbolNameList = []
        tsymbolList = []
        for j in range(B + 1 + 2 + 1) :
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
    po = []
    power = 1
    for i in range(len(symbolList[x])) :
        po.append(power)
        power *= 2
    po.append(power)
    ans = 0
    ans = sum ([(symbolList[x][i] + 1) * (0.5) * po[i] for i in range(len(symbolList[x]) - 4)])
    ans = sum([ans, (symbolList[x][len(symbolList[x]) - 4] + 1) * (0.5) * (maxa - pow(2,len(symbolList[x]) - 4) )])
    ans = sum([ans,-1*maxa])
    ans = sum([ans,(symbolList[x][-3] + 1) * (0.5) * maxa])
    ans = sum([ans,(symbolList[x][-2] + 1) * 0.5 * (maxa + 1)])
    return ans


def separate_numbers(input_string):
    # Split the input string by underscores
    parts = input_string.split('_')

    # Filter out non-numeric parts and convert them to integers
    numbers = [int(part) for part in parts if part.isdigit()]

    return numbers
def inner(i,symbolList):
    """TODO: Docstring for inner.

    :i: TODO
    :returns: TODO

    """
    ans = 0;
    ans = sum ([ 1 - ((symbolList[j][-3] + 1)*0.5) for j in range( i + 1,len(symbolList))])
    ans = sum([ans, -1*(1 - ((symbolList[i][-3] + 1)*0.5))])
    ans *= ((symbolList[i][-1] + 1)*(0.5))
    return ans

def remove_duplicates(list_of_lists):
    seen = set()
    unique_list_of_lists = []
    for inner_list in list_of_lists:
        # Convert the inner list to a tuple to make it hashable
        tuple_inner_list = tuple(inner_list)
        if tuple_inner_list not in seen:
            unique_list_of_lists.append(inner_list)
            seen.add(tuple_inner_list)
    return unique_list_of_lists


def norm(b):
    """TODO: Docstring for norm.

    :b: TODO
    :returns: TODO

    """
    return sum([b[i]*b[i] for i in range(len(b))])
def energy(sol, h, J):
    """TODO: Docstring for energy.

    :sol: TODO
    :symbolList: TODO
    :returns: TODO

    """ 
    ans = 0
    for i,value in h.items():
        ans += value * (sol[i])
    for (i,j),value in J.items() :
        ans += value *(sol[i]) * (sol[j])
    return ans







def f(list_of_lists):
    # Display the result
    list_of_lists = list_of_lists[:10]
    list_of_lists = np.matrix(list_of_lists)
    A = np.array(list_of_lists.T).tolist()
    maxa = 0
    # print(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            maxa = max(maxa, abs(A[i][j]))
    # print(maxa)  # range = [-maxa, maxa]
    maxa = 2 
    # print(maxa)
    floorloga_1 = np.log(float(maxa - 1))/np.log(2)
    floorloga_1 = int(floorloga_1)
    symbolNameList, symbolList = Creating_variables(len(A[0]), floorloga_1)
    #print(maxa)
    #print(floorloga_1)
    # print(len(symbolList[0]))
    #print(xieq(0,symbolList,maxa))
    print(1)

    b = [ sum([A[i][j]*xieq(j,symbolList,maxa) for j in range(len(A[0]))]) for i in range(len(A)) ]
    print(1)
    penalty = sum([inner(i,symbolList) for i in range(len(symbolList))])
    print(1)
    minimise = sum([b[i]*b[i] for i in range(len(b))])
    print(1)
    p = 1e3
    penalty = p*(1 + penalty)
    print(1)

    qubo_eq = sympy.expand(sum([100*minimise , penalty]))
    # qubo_eq = sympy.expand(minimise)
    print(2)
    result_dict_unreduced = qubo_eq.as_coefficients_dict()
    # print(result_dict_unreduced)
    result_dict = {
        (i, j): 0
        for i in range(len(symbolList)*(len(symbolList[0])))
        for j in range(i,len(symbolList)*(len(symbolList[0])))
    }
    for key in result_dict_unreduced.keys():
        if type(key) == sympy.core.power.Pow:
            # s_i**2 terms so always s_i**2 = 1 so it contributes to constant term
            pass
        elif type(key) == sympy.core.symbol.Symbol:
            # s_i terms
            one = separate_numbers(str(key))
            number = one[0]*(len(symbolList[0])) + one[1]
            result_dict[number,number] += result_dict_unreduced[
                key
            ]
        elif type(key) == sympy.core.mul.Mul:
            # cross terms
            one = separate_numbers(str(key.args[0]))
            two = separate_numbers(str(key.args[1]))
            ind2 = max(one[0]*(len(symbolList[0])) + one[1], two[0]*(len(symbolList[0])) + two[1])
            ind1 = min(one[0]*(len(symbolList[0])) + one[1], two[0]*(len(symbolList[0])) + two[1])
            result_dict[ind1, ind2] += result_dict_unreduced[key]
        elif type(key) == sympy.core.numbers.One:
            # constant
            pass
        else:
            print("ERROR : Possibily not QUBO")
    #print(result_dict)
    h = {}  # Dictionary for linear terms (h_i)
    J = {}  # Dictionary for quadratic terms (J_ij)
    # Set h_i values
    for i in range(len(symbolList)*(len(symbolList[0]))):  # Assuming n is the number of variables
        h[i] = result_dict[(i, i)]

    # Set J_ij values
    for (i, j), value in result_dict.items():
        if i != j:
            J[(i, j)] = value
    return h, J,qubo_eq,b,minimise,penalty,symbolNameList,symbolList
def simp(best_sol,simpeq,minimise,penalty,b,symbolNameList,symbolList):
    """TODO: Docstring for simp.

    :sol: TODO
    :returns: TODO

    """
    if type(best_sol) == list:
        ans = {}
        for i in range(len(best_sol)) :
            ans[i] = best_sol[i]
        best_sol = ans
    for u in best_sol : 
        simpeq = simpeq.subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_sol[u])
        minimise = minimise.subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_sol[u])
        penalty = penalty.subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_sol[u])
        b = [b[i].subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],best_sol[u]) for i in range(len(b))]
    return simpeq,minimise,penalty,b
h,J,qubo_eq,b,minimise,penalty,symbolNameList,symbolList = f(list_of_lists)
print(3)
"""
num_reads = 100 # Number of samples you want to obtain

# Create a DWaveSampler
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising(h, J, num_reads=num_reads, return_embedding=True)

best_solution = response.first.sample

print(best_solution)
simpeq = (qubo_eq)
btemp = (b)
"""

"""
print(simp(best_solution,simpeq,minimise,penalty,b,symbolNameList,symbolList))
_,_,_,bmin = simp(best_solution,simpeq,minimise,penalty,b,symbolNameList,symbolList)
"""
# Create IsingModel
ising_model = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

# Add linear terms
for v, h_v in h.items():
    ising_model.add_linear(v, h_v)

# Add quadratic terms
for (u, v), J_uv in J.items():
    ising_model.add_quadratic(u, v, J_uv)

# Graph corresponding to D-Wave 2000Q
qpu = DWaveSampler()
qpu_edges = qpu.edgelist
qpu_nodes = qpu.nodelist
# pprint(dir(qpu))
if qpu.solver.id == "DW_2000Q_6":
    print(qpu.solver.id)
    X = dnx.chimera_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    dnx.draw_chimera(X, node_size=1)
    print('Number of qubits=', len(qpu_nodes))
    print('Number of couplers=', len(qpu_edges))
elif qpu.solver.id == "Advantage_system4.1":
    print(qpu.solver.id)
    X = dnx.pegasus_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    dnx.draw_pegasus(X, node_size=1)
    print('Number of qubits=', len(qpu_nodes))
    print('Number of couplers=', len(qpu_edges))
DWavesampler = EmbeddingComposite(DWaveSampler())
DWaveSamples = DWavesampler.sample(bqm = ising_model, num_reads=2000,return_embedding=True)
embedding = DWaveSamples.info['embedding_context']['embedding']
if qpu.solver.id == "DW_2000Q_6":
  dnx.draw_chimera_embedding(X, embedding, node_size=2)
elif qpu.solver.id == "Advantage_system4.1":
  dnx.draw_pegasus_embedding(X, embedding, node_size=2)

simpeq = (qubo_eq)
btemp = (b)
qu,mini,pen,bmin = (simp(DWaveSamples.first.sample,simpeq,minimise,penalty,b,symbolNameList,symbolList))
# print(bmin)
# print(pen)
# print(mini)
# print(qu)
btemp = b
colec = []
cnt = 0
response = DWaveSamples
best_solution = response.first.sample
for uu in response: 
    cnt+=1 
    b = btemp
    if cnt >= 100:
        break
    for u in uu: 
        b = [b[i].subs(symbolNameList[int(u/(len(symbolList[0])))][u%(len(symbolList[0]))],uu[u]) for i in range(len(b))]
    colec.append(b)


colec = remove_duplicates(colec)
bmin = np.array(bmin)
colecnew = []
colec = colec[:100]
colec = sorted(colec, key=norm)
for i in range(len(colec) - 1):
    colecnew.append(np.array(colec[i]))
    colecnew.append(np.array(colec[i + 1]) - np.array(colec[i]))
    colecnew.append(np.array(colec[i + 1]) + np.array(colec[i]))
colec = colecnew
colec = colec[:100]
colec = sorted(colec, key=norm)
# print(len(colec))
for u in colec :
    if norm(bmin) > norm(bmin - u)  and (bmin.all() != u.all()):
        bmin = bmin-u
    elif norm(bmin) > norm(bmin + u) and (bmin.all() != -1*u.all()):
        bmin = bmin + u

for u in colec :
    if norm(bmin) > norm(bmin - u)  and (bmin.all() != u.all()):
        bmin = bmin-u
    elif norm(bmin) > norm(bmin + u) and (bmin.all() != -1*u.all()):
        bmin = bmin + u

for u in colec :
    if norm(bmin) > norm(bmin - u)  and (bmin.all() != u.all()):
        bmin = bmin-u
    elif norm(bmin) > norm(bmin + u) and (bmin.all() != -1*u.all()):
        bmin = bmin + u

for u in colec :
    if norm(bmin) > norm(bmin - u)  and (bmin.all() != u.all()):
        bmin = bmin-u
    elif norm(bmin) > norm(bmin + u) and (bmin.all() != -1*u.all()):
        bmin = bmin + u
print(bmin)

best = [-1 for i in range(len(best_solution))]
zero = best
b = btemp
for i in range(len(symbolList)):
    zero[i*len(symbolList[0]) + len(symbolList[0]) - 3] = 1
# print(energy(zero,h,J))
# print(energy(best_solution,h,J))
print(norm(bmin)**0.5)
