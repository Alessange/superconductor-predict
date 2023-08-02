import random

import torch
from torch.utils.data import Dataset
import json
import numpy as np
from Gen_atom import atom
import itertools


class PretrainDataset(Dataset):
    def __init__(self, data):
        if isinstance(data,
                      str):  # If data is a file path, load the data from the file
            with open(data, 'r') as file:
                self.data = json.load(file)
        else:  # If data is a list, use it directly
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]

        # Here, atom_sentence is a tensor where the first dimension corresponds to different atoms
        # and the second dimension is a concatenation of the atom's position and its properties.
        atom_sentence = torch.stack([torch.cat(
            (torch.Tensor(pos), torch.Tensor(atom(e).get_property()))) for
            pos, e in zip(positions, elements)])

        # Ensure target has 10 decimal places
        target = round(target, 10)
        return atom_sentence.unsqueeze(0), torch.Tensor([[[target]]])


class TestDataset(Dataset):
    def __init__(self, data):
        if isinstance(data,
                      str):  # If data is a file path, load the data from the file
            with open(data, 'r') as file:
                self.data = json.load(file)
        else:  # If data is a list, use it directly
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]

        # Here, atom_sentence is a tensor where the first dimension corresponds to different atoms
        # and the second dimension is a concatenation of the atom's position and its properties.
        atom_sentence = torch.stack([torch.cat(
            (torch.Tensor(pos), torch.Tensor(atom(e).get_property()))) for
            pos, e in zip(positions, elements)])

        # Ensure target has 10 decimal places
        target = round(target, 10)
        return atom_sentence.unsqueeze(0), torch.Tensor(
            [[[100]]]), torch.Tensor(
            [[[target]]])


class SuperDataset(Dataset):
    def __init__(self, data):
        if isinstance(data,
                      str):  # If data is a file path, load the data from the file
            with open(data, 'r') as file:
                self.data = json.load(file)
        else:  # If data is a list, use it directly
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]

        # shuffle indices
        indices = torch.randperm(len(elements))

        # Here, atom_sentence is a tensor where the first dimension corresponds to different atoms
        # and the second dimension is a concatenation of the atom's position and its properties.
        atom_sentence = torch.stack([torch.cat(
            (torch.Tensor(positions[i]),
             torch.Tensor(atom(elements[i]).get_property()))) for i in indices])

        # Ensure target has 10 decimal places
        target = round(target, 10)
        return atom_sentence.unsqueeze(0), torch.Tensor(
            [[[100]]]), torch.Tensor(
            [[[target]]])


class ExtendedPretrainDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as file:
                data = json.load(file)
        self.data = []
        for item in data:
            positions, elements, target = item
            num_permutations = 1
            if 7 > target >= 4:
                num_permutations = 3
            elif 10 > target >= 7:
                num_permutations = 5
            elif 15 > target >= 10:
                num_permutations = 16
            elif 20 > target >= 15:
                num_permutations = 30
            elif target > 20:
                num_permutations = 60
            for _ in range(num_permutations):
                indices = np.random.permutation(len(elements))
                shuffled_positions = [positions[i] for i in indices]
                shuffled_elements = [elements[i] for i in indices]
                self.data.append(
                    (shuffled_positions, shuffled_elements, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]
        atom_sentence = torch.stack([torch.cat(
            (torch.Tensor(pos), torch.Tensor(atom(e).get_property()))) for
            pos, e in zip(positions, elements)])
        target = round(target, 10)
        return atom_sentence.unsqueeze(0), torch.Tensor(
            [[[100]]]), torch.Tensor([[[target]]])


class PermutationEqualDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as file:
                data = json.load(file)

        self.data = []

        for item in data:
            positions, elements, target = item

            # Generate all permutations
            if target >= 5:
                permutations = list(
                    itertools.permutations(range(len(elements))))

                for perm in permutations:
                    shuffled_positions = [positions[i] for i in perm]
                    shuffled_elements = [elements[i] for i in perm]
                    self.data.append(
                        (shuffled_positions, shuffled_elements, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]
        atom_sentence = torch.stack([torch.cat(
            (torch.Tensor(pos), torch.Tensor(atom(e).get_property()))) for
            pos, e in zip(positions, elements)])

        target = round(target, 10)
        return atom_sentence.unsqueeze(0), torch.Tensor(
            [[[100]]]), torch.Tensor([[[target]]])


class TrainDataset(Dataset):
    def __init__(self, data):
        if isinstance(data,
                      str):  # If data is a file path, load the data from the file
            with open(data, 'r') as file:
                self.data = json.load(file)
        else:  # If data is a list, use it directly
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        positions, elements, target = self.data[idx]

        result = []
        lst = list(range(len(elements)))
        random.shuffle(lst)
        for i in lst:
            result.append(elements[i])
            result.append(positions[i])
        return result, torch.Tensor([[[0.00]]]), torch.Tensor([[[target]]])


# Note: all dataset suppose return position_feas, mask_target, original_target
if __name__ == "__main__":
    path1 = '/Users/huangshixun/Desktop/Transformer/superconductors/data/pre_train_new.json'
    path2 = '/Users/huangshixun/Desktop/Transformer/superconductors/data/test_pre_train_data.json'
    path3 = '/Users/huangshixun/Desktop/Transformer/superconductors/model_for_superconductor/data/superconductor_data.json'
    #dataset2 = PretrainDataset(path2)
    #dataset1 = PretrainDataset(path1)
    test_dataset3 = TrainDataset(path1)
    with open(path1, 'r') as f:
        dataset = json.load(f)
    z = max(len(data[1]) for data in dataset)
    print(z)
