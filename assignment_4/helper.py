import sympy
def read_input(filename):
    """Reads the input data of weights and values from a file

    :filename: filename where the data is present
    :returns: number_of_objects,Max_weight, value_weight

    """
    value_weight = []
    solution = []
    with open(filename, "r") as file:
        i = 0
        for line in file:
            if i == 0:
                number_of_object = line.strip()
                number_of_object = int(number_of_object)
            elif i == 1:
                Max_weight = line.strip()
                Max_weight = int(Max_weight)
            elif i >= 2 and i < 2 + number_of_object:
                temp = line.strip()
                value_weight.append([int(temp.split()[0]), int(temp.split()[1])])
            elif i >= 2 + number_of_object and i < 2 + 2 * number_of_object:
                temp = line.strip()
                solution.append(int(temp))
            i += 1
    B = 0
    while pow(2, B + 1) <= Max_weight:
        B += 1
    return (number_of_object, Max_weight, value_weight, B, solution)


def Creating_variables(number_of_object, B):
    symbolNameList = []
    symbolList = []
    for i in range(0, number_of_object + B + 1):
        name = "s_" + str(i)
        symbolNameList.append(name)
        symbolList.append(sympy.Symbol(name))

    return (symbolNameList, symbolList)




def Reduce_dictionary(result_dict_unreduced,number_of_object,B):
    """reduces the dictionay into required format

    :returns: result_dict

    """
    # initialising the dictionary that is to be displayed
    result_dict = {
        (i, j): 0
        for i in range(number_of_object + B + 1)
        for j in range(i, number_of_object + B + 1)
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
            ind2 = max(int(str(key.args[0])[2:]), int(str(key.args[1])[2:]))
            ind1 = min(int(str(key.args[0])[2:]), int(str(key.args[1])[2:]))
            result_dict[ind1, ind2] += result_dict_unreduced[key]
        elif type(key) == sympy.core.numbers.One:
            pass
        else:
            print("ERROR : Possibily not QUBO")

    return result_dict




def fill_h_J(result_dict,number_of_object,B):
    """fills the h and J dictionay so that it can be feed into quantum annealer

    :returns: h,J

    """
    h = {}  # Dictionary for linear terms (h_i)
    J = {}  # Dictionary for quadratic terms (J_ij)
    # Set h_i values
    for i in range(number_of_object + B + 1):  # Assuming n is the number of variables
        h[i] = result_dict[(i, i)]

    # Set J_ij values
    for (i, j), value in result_dict.items():
        if i != j:
            J[(i, j)] = value
    return (h, J)

def second_printer(left,right):
    """does the right and left justified printing
    """
    print(f"{left : <10}{right : >40}")
def Printer(solution,h,J,number_of_object,B,Max_weight,value_weight):
    """Decodes the solutions to give the max_value, lowest_energy and other data

    :solution: solution
    :returns: the decoded information
    """
    # second_printer("Best solution:", solution)
    value = 0
    weight = 0
    selected = []
    slack = 0
    for u in solution :
        if u >= number_of_object:
            if u != (number_of_object + B):
                slack += pow(2, (u - number_of_object)) * (solution[u] + 1) * (0.5)
            else:
                slack += (Max_weight - pow(2, B) + 1) * ((solution[u] + 1) * (0.5))
        elif solution[u] == -1:
            pass
        else:
            selected.append(value_weight[u])
            value += value_weight[u][0]
            weight += value_weight[u][1]

        # print("slack weight :", slack)
    print("selected [value, weight] pairs respectively:")
    print(selected)
    print()
    second_printer("total value selected:", value)
    second_printer("total weight selected:", weight)
    if slack + weight != Max_weight:
        # print("Warning : More qpu time required to reach the closer to global minima")
        slack = Max_weight - weight
    return (value,weight,slack,selected)

def Postprocessing(number_of_object,best_solution,value_weight,slack):
    """polynomial time postprocessing to get closer to global minimum

    """
    for i in range(number_of_object):
        if best_solution[i] == -1 and value_weight[i][1] <= slack:
            slack -= value_weight[i][1]
            best_solution[i] = 1
    for i in range(number_of_object):
        if best_solution[i] == 1:
            for j in range(number_of_object):
                if (
                    value_weight[j][0] > value_weight[i][0]
                    and value_weight[j][1] <= value_weight[i][1] + slack
                    and best_solution[j] == -1
                ):
                    slack -= value_weight[j][1] - value_weight[i][1]
                    best_solution[j] = 1
                    best_solution[i] = -1
                    break
    for i in range(number_of_object):
        if best_solution[i] == 1:
            for j in range(number_of_object):
                if (
                    value_weight[j][0] > value_weight[i][0]
                    and value_weight[j][1] <= value_weight[i][1] + slack
                    and best_solution[j] == -1
                ):
                    slack -= value_weight[j][1] - value_weight[i][1]
                    best_solution[j] = 1
                    best_solution[i] = -1
                    break
    for i in range(number_of_object):
        if best_solution[i] == 1:
            for j in range(number_of_object):
                if (
                    value_weight[j][0] > value_weight[i][0]
                    and value_weight[j][1] <= value_weight[i][1] + slack
                    and best_solution[j] == -1
                ):
                    slack -= value_weight[j][1] - value_weight[i][1]
                    best_solution[j] = 1
                    best_solution[i] = -1
                    break

    return best_solution

def actual_solution(value_weight,number_of_object,B,solution,Max_weight):
    """
    using the solution given in data file to find parametes like the enegy, weight and value
    """
    actual_weight = 0
    for i in range(number_of_object):
        solution[i] = (2 * solution[i] - 1) 
        if solution[i] == 1:
            actual_weight  += value_weight[i][1]
    solbit = []
    for sol in range(pow(2, B + 1)):
        solbit = []
        tsol = int(sol)
        while len(solbit) <= B:
            solbit.append(2 * (tsol & 1) - 1)
            tsol = int(tsol / 2)

        solbit_reverse = [solbit[len(solbit) - i - 1] for i in range(len(solbit))]
        solbit = solbit_reverse
        slack_weight_now = sum((pow(2, i) * (solbit[i] + 1) * (0.5)) for i in range(B))
        slack_weight_now = sum(
            [slack_weight_now, (Max_weight - pow(2, B) + 1) * ((solbit[B] + 1) * (0.5))]
        )
        if slack_weight_now == Max_weight - actual_weight:
            break
    for i in range(len(solbit)):
        solution.append(solbit[i])
    dict_solution = {}
    for i in range(len(solution)):
        dict_solution[i] = solution[i]
    return dict_solution


