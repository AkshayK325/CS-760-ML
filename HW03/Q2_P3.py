import pandas as pd
import numpy as np

df = pd.read_csv('hw3Data/emails.csv')

X = df.drop(columns=['Prediction']).values
X = X[:,1::]
y = df['Prediction'].values

X = np.hstack([np.ones((X.shape[0], 1)), X])

#sigmoid function
def sigmoidC(z):
    z = np.array(z.flatten(),dtype=float)
    sig = 1. / (1 + np.exp(-z.flatten()))
    return sig

#cost function
def cost_function(y, y_pred):
    m = len(y)
    return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

#gradient of the cost function
def gradient(X, y, y_pred):
    m = len(y)
    return (1/m) * np.dot(X.T, (y_pred - y))

#logistic regression with gradient descent
def logistic_regression(X, y, lr=0.1, epochs=1000):
    theta = np.zeros((X.shape[1], 1))
    for i in range(epochs):
        Xtheta = np.dot(X, theta)
        y_pred = sigmoidC(Xtheta)
        gradients = gradient(X, y, y_pred)
        theta = theta - lr * gradients.reshape((-1,1))
        # if np.linalg.norm(gradients.reshape((-1,1))) < 8:
        #     break;
    return theta

# 5-Fold CV
folds = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
results = []
j=0;
for fold in folds:
    print("Running the fold number = ",j+1)
    test_X = X[fold[0]:fold[1]]
    test_y = y[fold[0]:fold[1]]
    train_X = np.concatenate((X[:fold[0]], X[fold[1]:]), axis=0)
    train_y = np.concatenate((y[:fold[0]], y[fold[1]:]), axis=0)

    # logistic regression
    theta = logistic_regression(train_X, train_y, lr=0.001, epochs=500)

    # Predict
    y_pred = sigmoidC(np.dot(test_X, theta))
    predictions = (y_pred >= 0.5).astype(int)
    predictions= np.array(predictions)

    #Calculate metrics
    accuracy = np.mean(predictions == test_y)
    precision = np.sum((predictions == 1) & (test_y == 1)) / np.sum(predictions == 1)
    recall = np.sum((predictions == 1) & (test_y == 1)) / np.sum(test_y == 1)

    results.append((accuracy, precision, recall))
    j=j+1
    
# results
for idx, (accuracy, precision, recall) in enumerate(results, 1):
    print(f"Fold {idx}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("-" * 50)
