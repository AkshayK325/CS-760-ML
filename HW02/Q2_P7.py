import numpy as np
import matplotlib.pyplot as plt

from DecisionTreeClassifierCustom import DecisionTree
#text file data
D = np.loadtxt('HomeworkData/Dbig.txt', dtype=float)
#shuffle the array randomly
numRand = np.random.permutation(np.arange(10000))
D = D[numRand, :]

setSize = np.array([8192, 2048, 512, 128, 32])

dataTrSets = []
dataTestSets = D[setSize[0]:-1, :]
for i in setSize:
    dataTrSets.append(D[0:i, :])

# initialize lists
n_nodes_list = []
error_rate_list = []
FS = 4 # font size

# Build decision trees
for data in dataTrSets:
    Xtr = data[:, 0:2]
    ytr = data[:, -1]

    tree = DecisionTree(random_seed=42)
    tree.fit(np.column_stack((Xtr, ytr))) 
    
    n_nodes = tree.count_nodes()
    # plot
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.title(f'Decision Tree with {n_nodes} Nodes')
    tree.plot_tree(feature_names=['Feature 1', 'Feature 2'], class_names=['Class 0', 'Class 1'],fontSize=FS)
    plt.show()
    FS = FS + 1
    Xtest = dataTestSets[:, 0:2]
    ytest = dataTestSets[:, -1]
    
    #make predictions
    predictions = [tree.predict(instance) for instance in Xtest]
    error_rate = 1 - np.mean(predictions == ytest)
    
    n_nodes_list.append(n_nodes)
    error_rate_list.append(error_rate)

# Display the results
print("Set size =", setSize)
print("Node List =", n_nodes_list)
print("error List =", error_rate_list)

#plot the results
plt.figure(1)
plt.plot(setSize, error_rate_list, marker='o', color='red', label='Error Rate')
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
