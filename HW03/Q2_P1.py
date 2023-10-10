import numpy as np
import matplotlib.pyplot as plt

#text file data
D = np.loadtxt('hw3Data/D2z.txt', dtype=float)

X_train = D[:, :2]  # The first two columns are features
y_train = D[:, 2]   # The third column is the label

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def one_nearest_neighbor(x, X_train, y_train):
    distances = [euclidean_distance(x, x_train) for x_train in X_train]
    nearest_neighbor_index = np.argmin(distances)
    return y_train[nearest_neighbor_index]

# grid test points
x = np.arange(-2, 2.1, 0.1)
y = np.arange(-2, 2.1, 0.1)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

#1NN
labels = [one_nearest_neighbor(point, X_train, y_train) for point in grid_points]
labels = np.array(labels).reshape(xx.shape)

#Plot
plt.contourf(xx, yy, labels, alpha=0.5, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='m', marker='o', cmap='coolwarm', s=50, label="Training Data Points")
plt.title("1NN Predictions on 2D Grid")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar()
plt.legend(loc='upper right')
plt.show()
