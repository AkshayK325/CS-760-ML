import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
data2D = pd.read_csv('data/data2D.csv', header=None).values
data1000D = pd.read_csv('data/data1000D.csv', header=None).values

# norm error
def error_norm(recon, orig):
    return np.linalg.norm(recon - orig, 'fro')**2

# Buggy PCA (no mean subtraction)
def buggy_pca(data, d):
    U, S, Vt = svd(data, full_matrices=False)
    Z = U[:, :d] * S[:d]
    A = Vt[:d, :]
    reconstruction = Z @ A
    return Z, A, reconstruction

# Demeaned PCA
def demeaned_pca(data, d):
    mean = np.mean(data, axis=0)
    demeaned_data = data - mean
    pca = PCA(n_components=d)
    Z = pca.fit_transform(demeaned_data)
    A = pca.components_
    reconstruction = pca.inverse_transform(Z) + mean
    return Z, A, reconstruction

# Normalized PCA
def normalized_pca(data, d):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    pca = PCA(n_components=d)
    Z = pca.fit_transform(normalized_data)
    A = pca.components_
    reconstruction = pca.inverse_transform(Z) * std + mean
    return Z, A, reconstruction

# DRO implementation would be more involved as it requires an optimization procedure
# This is a placeholder function for DRO
def dro(data, d):
    iterations = 100
    # Initialize b as the mean of the data
    b = np.mean(data, axis=0)
    data_centralized = data - b

    # Initialize A using SVD
    U, S, Vt = svd(data_centralized, full_matrices=False)
    A = Vt.T[:, :d]

    # Initialize Z
    Z = np.dot(data_centralized, A)

    # Iterate to optimize Z and A
    for _ in range(iterations):  # Define the number of iterations
        # Update Z given A, with constraints
        Z = np.dot(data_centralized, A)
        Z -= np.mean(Z, axis=0)  # Ensure Z has zero mean
        Uz, Sz, Vzt = svd(Z, full_matrices=False)
        Z = Uz @ np.eye(d)  # Enforce identity covariance
        
        # Update A given Z
        A = np.linalg.solve(Z.T @ Z, Z.T @ data_centralized).T

    # Recalculate b since we have changed Z
    b = np.mean(data - np.dot(Z, A.T), axis=0)

    # Reconstruction from Z
    reconstruction = np.dot(Z, A.T) + b
    
    return Z, A, reconstruction


def applyToDataset(data,d=1):
    # Apply the methods to the 2D dataset
    Z_buggy, A_buggy, recon_buggy = buggy_pca(data, d)
    Z_demeaned, A_demeaned, recon_demeaned = demeaned_pca(data, d)
    Z_normalized, A_normalized, recon_normalized = normalized_pca(data, d)
    Z_dro, A_dro, recon_dro = dro(data, d)
    
    # Compute reconstruction errors
    error_buggy = error_norm(recon_buggy, data)
    error_demeaned = error_norm(recon_demeaned, data)
    error_normalized = error_norm(recon_normalized, data)
    error_dro = error_norm(recon_dro, data)
    
    recons = {
        'buggy': recon_buggy,
        'demeaned': recon_demeaned,
        'normalized': recon_normalized,
        'dro': recon_dro
    }

    errors = {
        'buggy': error_buggy,
        'demeaned': error_demeaned,
        'normalized': error_normalized,
        'dro': error_dro
    }
    return data,recons,errors


inputData = 'data2D'

if inputData == 'data2D':
    d=1;
else:
    d=499;
    
data = globals()[inputData]

# finding knee point
# U, S, Vt = np.linalg.svd(data.T, full_matrices=False)
# plt.figure(figsize=(10, 5))
# plt.plot(S, 'o-')
# plt.title('Singular Values ("Scree Plot")')
# plt.xlabel('Component number')
# plt.ylabel('Singular value')
# plt.show(block=True)

data,recons,errors = applyToDataset(data,d)

# Print out the reconstruction errors
print('Buggy PCA Error:', errors['buggy'])
print('Demeaned PCA Error:' ,errors['demeaned'])
print('Normalized PCA Error:',errors['normalized'])
print('DRO Error:', errors['dro'])

# Plot the original and reconstructed points for each PCA variant
plt.figure(figsize=(15, 5))


# Buggy PCA
plt.subplot(1, 4, 1)
plt.scatter(data[:, 0], data[:, 1], label=inputData, alpha=0.5)
plt.scatter(recons['buggy'][:, 0], recons['buggy'][:, 1], label='Buggy PCA Reconstruction', alpha=0.5)
plt.title('Buggy PCA')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

# Demeaned PCA
plt.subplot(1, 4, 2)
plt.scatter(data[:, 0], data[:, 1], label=inputData, alpha=0.5)
plt.scatter(recons['demeaned'][:, 0], recons['demeaned'][:, 1], label='Demeaned PCA Reconstruction', alpha=0.5)
plt.title('Demeaned PCA')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

# Normalized PCA
plt.subplot(1, 4, 3)
plt.scatter(data[:, 0], data[:, 1], label=inputData, alpha=0.5)
plt.scatter(recons['normalized'][:, 0], recons['normalized'][:, 1], label='Normalized PCA Reconstruction', alpha=0.5)
plt.title('Normalized PCA')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

# DRO
plt.subplot(1, 4, 4)
plt.scatter(data[:, 0], data[:, 1], label=inputData, alpha=0.5)
plt.scatter(recons['dro'][:, 0], recons['dro'][:, 1], label='DRO', alpha=0.5)
plt.title('DRO')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

plt.tight_layout()
plt.show()