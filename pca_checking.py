from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt

# Load example dataset
digits = load_digits()

# Define a list of kernels to test
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Apply PCA with each kernel and plot results
for kernel in kernels:
    # Apply kernel PCA
    kpca = KernelPCA(n_components=2, kernel=kernel, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(digits.data)

    # Apply standard PCA for comparison
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digits.data)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, edgecolor='none', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('PCA')

    plt.subplot(122)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=digits.target, edgecolor='none', alpha=0.5)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title(f'Kernel PCA ({kernel} kernel)')

    plt.tight_layout()
    plt.show()
