import pandas as pd
import numpy as np

df = pd.read_csv('hw3Data/emails.csv')

X = df.drop(columns=['Prediction']).values
X = X[:,1::]
y = df['Prediction'].values


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def one_nn(train_X, train_y, test_point):
    distances = [euclidean_distance(train_point, test_point) for train_point in train_X]
    nearest = np.argmin(distances)
    return train_y[nearest]

#5-Fold CV
folds = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
results = []
j=0;
for fold in folds:
    print("Running the fold number = ",j+1)
    test_X = X[fold[0]:fold[1]]
    test_y = y[fold[0]:fold[1]]
    train_X = np.concatenate((X[:fold[0]], X[fold[1]:]), axis=0)
    train_y = np.concatenate((y[:fold[0]], y[fold[1]:]), axis=0)

    # Predict
    predictions = [one_nn(train_X, train_y, test_point) for test_point in test_X]
    
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
