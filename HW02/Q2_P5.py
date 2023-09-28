import numpy as np
from math import log2
import matplotlib.pyplot as plt
from DecisionTreeClassifierCustom import DecisionTree
# text file data
data = np.loadtxt('HomeworkData/D1.txt', dtype=float)

tree = DecisionTree()

# Fit the decision tree D1
tree.fit(data)
    
n_nodes = tree.count_nodes()
    
# plot
plt.figure(figsize=(12, 8))
plt.axis('off')
plt.title(f'Decision Tree with {n_nodes} Nodes')
tree.plot_tree(feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'],fontSize=18)
plt.show()

data = np.loadtxt('HomeworkData/D2.txt', dtype=float)

tree = DecisionTree()

# Fit the decision tree D2
tree.fit(data)
    
n_nodes = tree.count_nodes()

# plot
plt.figure(figsize=(12, 8))
plt.axis('off')
plt.title(f'Decision Tree with {n_nodes} Nodes')
tree.plot_tree(feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'],fontSize=7)
plt.show()
