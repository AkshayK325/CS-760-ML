import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
from scipy.optimize import linear_sum_assignment

def evaluate_kmeans(data, true_labels,n_restarts):
    best_score = float('inf')
    best_accuracy = 0
    best_kmeans = None
    for _ in range(n_restarts):  # Try multiple restarts to avoid local minima
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=None)
        kmeans.fit(data)
        score = kmeans.inertia_  # This is the K-means objective
        if score < best_score:
            best_score = score
            best_kmeans = kmeans
            
            # Find the best accuracy
            predicted_labels = kmeans.labels_
            indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
            map_labels = [true_labels[index] for index in indices]
            aligned_labels = [map_labels[predicted] for predicted in predicted_labels]
            best_accuracy = np.mean(np.array(true_labels) == np.array(aligned_labels))

    return best_score, best_accuracy
def evaluate_gmm(data, true_labels,n_restarts):
    best_score = float('-inf')
    best_accuracy = 0
    best_gmm = None
    for _ in range(n_restarts):
        gmm = GaussianMixture(n_components=3, n_init=10, max_iter=300, random_state=None)
        gmm.fit(data)
        score = -gmm.score(data) * len(data)  # This is the GMM objective (negative log likelihood)
        if score > best_score:
            best_score = score
            best_gmm = gmm

            # Find the best accuracy
            predicted_labels = gmm.predict(data)
            clusters = np.array([np.mean(data[predicted_labels == i], axis=0) for i in range(3)])
            indices,_ = pairwise_distances_argmin_min(clusters, data)
            map_labels = [true_labels[index] for index in indices]
            aligned_labels = [map_labels[predicted] for predicted in predicted_labels]
            best_accuracy = np.mean(np.array(true_labels) == np.array(aligned_labels))

    return best_score, best_accuracy 
# Define the means and covariances for the Gaussians P_a, P_b, P_c
means = {
    'a': np.array([-1, -1]),
    'b': np.array([1, -1]),
    'c': np.array([0, 1])
}
covariances = {
    'a': np.array([[2, 0.5], [0.5, 1]]),
    'b': np.array([[1, -0.5], [-0.5, 2]]),
    'c': np.array([[1, 0], [0, 2]])
}

# Generate datasets
datasets = {}
for sigma in [0.5, 1, 2, 4, 8]:
    data = []
    labels = []
    for label, mean in means.items():
        cov = sigma * covariances[label]
        samples = np.random.multivariate_normal(mean, cov, 100)
        data.append(samples)
        labels.extend([label] * 100)
    datasets[sigma] = (np.vstack(data), labels)


# Assuming the functions `evaluate_kmeans` and `evaluate_gmm` exist and return
# the objective value and the clustering accuracy.

kmeans_objectives = []
kmeans_accuracies = []
gmm_objectives = []
gmm_accuracies = []
n_restarts = 10

for sigma, (data, true_labels) in datasets.items():
    obj_kmeans, acc_kmeans = evaluate_kmeans(data, true_labels,n_restarts)
    obj_gmm, acc_gmm = evaluate_gmm(data, true_labels,n_restarts)
    
    kmeans_objectives.append(obj_kmeans)
    kmeans_accuracies.append(acc_kmeans)
    gmm_objectives.append(obj_gmm)
    gmm_accuracies.append(acc_gmm)

# Plotting the results
sigmas = [0.5, 1, 2, 4, 8]
print("Sigma values = ",sigmas)
print("K-mean Objective = ",kmeans_objectives)
print("GMM Objective = ",gmm_objectives)
print("K-mean accuracies  = ",kmeans_accuracies)
print("GMM accuracies = ",gmm_accuracies)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(sigmas, kmeans_objectives, label='K-means')
plt.plot(sigmas, gmm_objectives, label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Objective Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sigmas, kmeans_accuracies, label='K-means')
plt.plot(sigmas, gmm_accuracies, label='GMM')
plt.xlabel('Sigma')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
