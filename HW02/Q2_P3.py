import numpy as np
import matplotlib.pyplot as plt

from DecisionTreeClassifierCustom import DecisionTree

def entropy(data):
    n = len(data)
    if n == 0:
        return 0
    count0 = np.sum(data[:, -1] == 0)
    count1 = n - count0
    p0 = count0 / n
    p1 = count1 / n
    if p0 == 0 or p1 == 0:
        return 0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

def information_gain( data, split):
    n = len(data)
    left_mask = data[:, split[0]] >= split[1]
    right_mask = ~left_mask
    entropy_parent = entropy(data)
    print("Entropy Value=",entropy_parent)
    entropy_left = entropy(data[left_mask])
    entropy_right = entropy(data[right_mask])
    weight_left = np.sum(left_mask) / n
    weight_right = np.sum(right_mask) / n
    gain = entropy_parent - (weight_left * entropy_left + weight_right * entropy_right)
    return gain

def find_best_split(data):
    best_split = None
    best_gain = -1
    num_features = data.shape[1] - 1  
    for j in range(num_features):
        values = np.unique(data[:, j])
        for c in values:
            split = (j, c)
            gain = information_gain(data, split)
            print("Split and Gain = ",split,gain)
            if gain > best_gain:
                best_gain = gain
                best_split = split
    return best_split, best_gain

#text file data load
D = np.loadtxt('HomeworkData/Druns.txt', dtype=float)

Xtr = D[:, 0:2]
ytr = D[:, -1]

best_split, best_gain = find_best_split(D)

#output the best split and its information gain ratio
print("Best Split:", best_split)
print("Information Gain Ratio:", best_gain)


