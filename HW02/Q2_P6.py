import numpy as np
from math import log2
import matplotlib.pyplot as plt
from DecisionTreeClassifierCustom import DecisionTree
#text file data
data = np.loadtxt('HomeworkData/D1.txt', dtype=float)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of D1')
plt.show()

tree = DecisionTree()

# Fit the decision tree with your dataset
tree.fit(data)
    
n_nodes = tree.count_nodes()

# create a mesh grid
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Xtest = np.vstack((xx.reshape(-1),yy.reshape(-1))).T

Z1=xx.reshape(-1)*0;i=0
# predict class labels for each point
for instance in Xtest:
    Z1[i] = tree.predict(instance) 
    i=i+1
    
Z1 = Z1.reshape(xx.shape)

# plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z1, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(data[:, 0], data[:, 1], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for D1')
plt.show()


data = np.loadtxt('HomeworkData/D2.txt', dtype=float)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of D2')
plt.show()
tree = DecisionTree()

#fit the decision tree
tree.fit(data)
    
n_nodes = tree.count_nodes()

Z1=xx.reshape(-1)*0;i=0
#predict class labels for each point
for instance in Xtest:
    Z1[i] = tree.predict(instance) 
    i=i+1
    
Z1 = Z1.reshape(xx.shape)

# plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z1, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(data[:, 0], data[:, 1], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for D2')
plt.show()
