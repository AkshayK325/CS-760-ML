import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('hw3Data/emails.csv')

X = df.drop(columns=['Prediction']).values
X = X[:,1::]
y = df['Prediction'].values

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

#kNN function
def kNN(X_train, y_train, test_point, k):
    distances = [euclidean_distance(train_point, test_point) for train_point in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    
    # most common label in the kNN
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    most_common_label = unique_labels[np.argmax(counts)]
    return most_common_label

#accuracy calculation
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 5-Fold CV
folds = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
k_values = [1, 3, 5, 7, 10]
results = []

for k in k_values:
    accuracies = []
    j=0;
    for fold in folds:
        print("Running the fold number = ",j+1)
        test_X = X[fold[0]:fold[1]]
        test_y = y[fold[0]:fold[1]]
        train_X = np.concatenate((X[:fold[0]], X[fold[1]:]), axis=0)
        train_y = np.concatenate((y[:fold[0]], y[fold[1]:]), axis=0)
        
        predictions = [kNN(train_X, train_y, test_point, k) for test_point in test_X]
        accuracies.append(accuracy(test_y, predictions))
        j=j+1
    results.append(np.mean(accuracies))

#results
plt.plot(k_values, results, marker='o')
plt.xlabel('k')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy vs. k')
plt.show()

# Print
for k, accuracy in zip(k_values, results):
    print(f"k = {k}: Average Accuracy = {accuracy:.4f}")
