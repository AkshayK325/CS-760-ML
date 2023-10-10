import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('hw3Data/emails.csv')

X = df.drop(columns=['Prediction']).values
X = X[:,1::]
y = df['Prediction'].values

# Split data
train_X = X[:4001]
train_y = y[:4001]
test_X = X[4001:]
test_y = y[4001:]

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# kNN
def kNN(X_train, y_train, test_points, k):
    predictions = []
    for test_point in test_points:
        distances = [euclidean_distance(train_point, test_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(most_common_label)
    return predictions

# Logistic Regression
def logistic_regression(X, y, num_iterations, learning_rate):
    n_samples, n_features = X.shape
    # initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0

    #GD
    for _ in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        linear_model = np.array(linear_model.flatten(),dtype=float)

        predicted_y = 1 / (1 + np.exp(-linear_model))
        
        # Compute gradients
        dw = (1/n_samples) * np.dot(X.T, (predicted_y - y))
        db = (1/n_samples) * np.sum(predicted_y - y)
        
        # print(dw.shape,db,weights.shape,bias)
        # Update weights and bias
        weights -= learning_rate * np.array(dw.flatten(),dtype=float)
        bias -= learning_rate * db
    
    return weights, bias

def predict_logistic(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    linear_model = np.array(linear_model.flatten(),dtype=float)

    return 1 / (1 + np.exp(-linear_model))


# kNN to make predictions
knn_predictions = kNN(train_X, train_y, test_X, 5)

# Logistic Regression to make predictions
weights, bias = logistic_regression(train_X, train_y, num_iterations=500, learning_rate=0.001)
logistic_predictions = predict_logistic(test_X, weights, bias)

# ROC Curve
from sklearn.metrics import roc_curve

knn_probs = [1 if i == 1 else 0 for i in knn_predictions]
fpr_knn, tpr_knn, _ = roc_curve(test_y, knn_probs)
fpr_logistic, tpr_logistic, _ = roc_curve(test_y, logistic_predictions)

plt.figure()
plt.plot(fpr_knn, tpr_knn, label="kNN")
plt.plot(fpr_logistic, tpr_logistic, label="Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
