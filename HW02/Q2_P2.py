import numpy as np
import matplotlib.pyplot as plt

from DecisionTreeClassifierCustom import DecisionTree
#text file data converted to integer data type
D = np.loadtxt('HomeworkData/Q2_2Data.txt', dtype=float)

Xtr = D[:, 0:2]
ytr = D[:, -1]

# fit the decision tree and plot
tree = DecisionTree(random_seed=42)
tree.fit(np.column_stack((Xtr, ytr))) 
n_nodes = tree.count_nodes()

plt.figure(figsize=(12, 8))
plt.axis('off')
tree.plot_tree(feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'],fontSize=10)
plt.title(f'Decision Tree with {n_nodes} Nodes')
plt.show()

