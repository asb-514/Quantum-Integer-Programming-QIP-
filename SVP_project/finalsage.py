from sage.interfaces.four_ti_2 import four_ti_2
import numpy as np
# Open the file for reading
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

# Display the result
A = np.array(list_of_lists).tolist()
