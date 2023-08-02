import numpy as np
import json


def vector_in_new_basis(vector, basis):
    return np.dot(np.linalg.inv(basis), vector)


path1 = "path of dataset"
new_list = []
with open(path1, 'r') as file:
    data = json.load(file)
    for value in data:
        if value['Tc'] <= 300:
            basis = np.array(value['atoms']['lattice_mat'])
            cow = []
            result = []
            for vector in value['atoms']['coords']:
                new_v = np.array(vector)
                result1 = vector_in_new_basis(new_v, basis)
                cow.append(result1.tolist())  # Convert numpy.ndarray to list
            result.append(cow)
            result.append(value['atoms']['elements'])
            result.append(value['Tc'])
            new_list.append(result)
path2 = "expect dataset directory and format"
with open(path2,'w') as file:
    json.dump(new_list, file, indent=2)
