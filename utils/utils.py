import numpy as np
import matplotlib.pyplot as plt

def create_dataset(N,D=2,K=2):
    """
    Generates a synthetic dataset of K classes each having N samples with D features. 
    The generated dataset exhibits a spiral pattern for each class.
    
    Parameters:
        N (int): Number of samples per class.
        D (int, optional): Number of features per sample. Defaults to 2.
        K (int, optional): Number of classes in the dataset. Defaults to 2.
    
    Returns:
        X (numpy.ndarray): Data matrix where each row corresponds to a single example.
        y (numpy.ndarray): Array of class labels for each example in X.
    """
    
    X = np.zeros((N * K, D)) # Data matrix (each row = single example)
    y = np.zeros(N * K) # Class labels
    
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N) # Radius
        # Theta: Generates a linear array of angles with a slight random noise
        t = np.linspace( j * 4, (j + 1) * 4, N ) + np.random.randn(N) * 0.2 # Theta
        # Transforming polar coordinates (r, theta) to Cartesian coordinates (x, y)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j # Assign class label
        
    # Visualization of the dataset
    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
    y[y == 0] -= 1
    
    return X, y


# Explanation of the Code
# Function Definition: The function create_dataset is defined with three parameters: N, D, and K, where N is the number of samples per class, D is the dimensionality of each sample (default is 2 for 2D data), and K is the number of classes.
# 
# Data Matrix and Labels: Two numpy arrays, X and y, are initialized to store the dataset and its corresponding labels. X has a shape of (N*K, D) to accommodate all samples from all classes, and y is a 1D array with length N*K.
# 
# Data Generation Loop: The loop iterates over each class j. For each class, it generates linearly spaced values for radius r and angle theta (t). The angle has a random noise added to it to create a spiral pattern. Polar coordinates are transformed to Cartesian coordinates and assigned to X. The corresponding labels are stored in y.
# 
# Visualization: The dataset is visualized using a scatter plot with plt.scatter, coloring each class differently for visual distinction.
# 
# Return Statement: Finally, the function returns the generated data matrix X and the label array y.

def plot_contour(X, y, model):
    """
    Plots the decision boundaries of an SVM classifier along with the data points.

    Parameters:
        X (numpy.ndarray): The input features, where each row represents a sample and each column represents a feature.
        y (numpy.ndarray): The class labels for each sample in X.
        model (classifier with predict method): A trained model classifier used to predict the class labels.

    This function creates a high-resolution contour plot representing the decision boundaries of the SVM classifier. It also overlays the original data points on the plot, coloring them based on their actual class labels for easy comparison.
    """
    # plot the resulting classifier
    h = 0.01  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Plot the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
