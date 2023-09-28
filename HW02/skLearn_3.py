import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt

#text file data converted to integer data type
D = np.loadtxt('HomeworkData/Dbig.txt', dtype=float)
#shuffle the array randomly
numRand = np.random.permutation(np.arange(10000))
D = D[numRand,:]

setSize = np.array([8192,2048,512,128,32])

dataTrSets = []
dataTestSets = D[setSize[0]:-1,:]
for i in setSize:
    dataTrSets.append(D[0:i,:])

#initialize lists to store results
n_nodes_list = []
error_rate_list = []
tree = DecisionTreeClassifier(random_state=42)

#build decision trees and fit
for data in dataTrSets:
    Xtr = data[:,0:2]
    ytr = data[:,-1]
    
    tree = tree.fit(Xtr, ytr)
    n_nodes = tree.tree_.node_count
    
    # plot the decision tree
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    plot_tree(tree, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'])
    plt.title(f'Decision Tree with {n_nodes} Nodes')
    plt.show()
    
    Xtest = dataTestSets[:,0:2]
    ytest = dataTestSets[:,-1]
    error_rate = 1 - tree.score(Xtest, ytest)
    n_nodes_list.append(n_nodes)
    error_rate_list.append(error_rate)

# display the results
print("Set size =",setSize)
print("Node List =",n_nodes_list)
print("error List =",error_rate_list)

# plot the results
plt.figure(1)
plt.plot(setSize, error_rate_list, marker='o',color='red', label='Error Rate')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (log scale)')
plt.ylabel('Error Rate (log scale)')
plt.title('Error Rate vs. Number of Samples')
plt.legend()
plt.grid()
plt.show(block=False)

plt.figure(2)
plt.plot(setSize, n_nodes_list, marker='o', label='Number of Nodes')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (log scale)')
plt.ylabel('Number of Nodes (log scale)')
plt.title('Number of Nodes vs. Number of Samples')
plt.legend()
plt.grid(True)
plt.show()


