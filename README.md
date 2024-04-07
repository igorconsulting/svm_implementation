# Support Vector Machine (SVM) Implementation Using cvxopt

<p align="center">
  <img src="https://github.com/igorconsulting/svm_implementation/blob/main/image/datamodelimg.png" alt="Data Visualization" title="View of Data Distribution">
</p>



This project outlines an implementation of a Support Vector Machine (SVM), a versatile machine learning algorithm used for classification, regression, and outlier detection. The implementation leverages the cvxopt library for convex optimization, numpy for numerical operations, and includes the use of various kernel functions to handle both linear and non-linear datasets.

This implementation demonstrates the adaptability of SVMs to both linear and non-linear datasets through the use of different kernel functions. The optimization of decision boundaries via convex optimization underscores SVMs' robust classification capabilities.

## Libraries and Utility Functions

- **cvxopt**: Utilized for solving the quadratic programming problem inherent in training SVMs.
- **numpy**: Essential for efficient array and matrix operations.
- **utils**: Contains utility functions like `create_dataset` for generating datasets and `plot_contour` for visualizing decision boundaries.

## Kernel Functions

Kernel functions are critical in SVMs for computing the similarity between vectors, allowing for the transformation of data into higher-dimensional spaces.

- **Linear Kernel**:
$$K(x, z) = x · z^T$$
This kernel is suitable for linearly separable data, relying on the dot product between vectors.
- **Polynomial Kernel**:
$$ K(x, z) = (1 + x · z^T)^p $$
By transforming the input space into a polynomial feature space, this kernel enables non-linear decision boundaries.
- **Gaussian (RBF) Kernel**:

$$ K(x, z) = exp(-\frac{||x - z||^2}{2 \sigma^2}) $$

Ideal for non-linear data, this kernel projects inputs into an infinite-dimensional space, facilitating linear separability.

## SVM Class

The `SVM` class encapsulates the SVM model, including methods for training and prediction, as well as for parameter extraction.

### Initialization

- **`__init__(self, kernel=gaussian, C=1)`**: Initializes the model with a specified kernel and regularization parameter `C`.

### Fit Method

- **`fit(self, X, y)`**: Trains the model on the dataset `(X, y)`, computing the kernel matrix and solving the quadratic programming problem to find optimal Lagrange multipliers. This process involves setting up constraints based on the regularization parameter `C`.

### Predict Method

- **`predict(self, X)`**: For new inputs `X`, the method predicts class labels. It identifies support vectors, calculates the weight vector `w` and bias `b`, and uses these to determine the class based on the sign of `w · x + b`.

### get_parameters Method

- **`get_parameters(self, alphas)`**: Extracts the model parameters (`w` and `b`) and identifies support vectors, using the Lagrange multipliers `alphas`.

## Training Process

The SVM training involves a quadratic programming problem aimed at maximizing the margin between decision boundaries and support vectors, subject to classification accuracy constraints influenced by the regularization parameter `C`.

# Solving the SVM Quadratic Programming Problem

At the core of the Support Vector Machine (SVM) training process is the optimization of a quadratic programming (QP) problem. This section aims to elucidate the step-by-step process of solving this QP problem, which is fundamental in determining the SVM's decision function that best separates the classes in the dataset.

## The Quadratic Programming Problem

For the softmax margin SVM, recall that the optimization problem can be expressed as

$$\begin{align*}
\text{maximize}_{\alpha} \quad & \sum_{i} \alpha_i - \frac{1}{2} \alpha^T H \alpha \\
\text{subject to} \quad       & 0 \leq \alpha_i \leq C \\
\;& \sum_{i} {\alpha}_i y^{(i)} = 0
\end{align*}$$ 

Which can be written in standard form as

$$\begin{align*}
\text{minimize}_\alpha \quad & \frac{1}{2} \alpha^T H \alpha - 1^T \alpha  \\
\text{subject to} \quad & -\alpha_i \leq 0 \\
& \alpha_i \leq C \\
& y^T \alpha = 0
\end{align*}$$

The QP problem in the context of SVMs is formulated as follows:

- **Objective Function**: 
  $$\min_{\alpha}\frac{1}{2} \mathbf{\alpha}^T P \mathbf{\alpha} - \mathbf{q}^T \mathbf{\alpha} $$

- **Subject to Constraints**:
  - $ G\mathbf{\alpha} \leq \mathbf{h} $
  - $ A\mathbf{\alpha} = \mathbf{b} $

where:
- $P$ is a matrix derived from the kernel function applied to the training data.
- $\mathbf{q}$ is a vector with all elements set to -1.
- $G$ and $\mathbf{h}$ enforce the box constraints on the Lagrange multipliers $\alpha_i$, ensuring $0 \leq \alpha_i \leq C$.
- $A$ and $\mathbf{b}$ ensure the sum of the product of Lagrange multipliers and their corresponding labels is zero, aligning with the Karush-Kuhn-Tucker (KKT) conditions for optimality.

## Step-by-Step Breakdown

### Kernel Matrix Calculation
The initial step involves computing the kernel matrix $K$, with each element $K_{ij}$ being the result of the kernel function applied to the data point pairs $(x_i, x_j)$. This matrix captures the data's geometry in the feature space.

### Formulating the Objective Function
The matrix $P$ is constructed as the Hadamard (element-wise) product of $y_iy_j$ and the kernel matrix $K$, embedding the target labels within the optimization problem, thereby facilitating classification.

### Defining the Constraints
The constraints are twofold:
1. Box constraints $0 \leq \alpha_i \leq C$ are specified by $G$ and $\mathbf{h}$, with $C$ being the regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error.
2. The equality constraint $\sum_{i=1}^{m} \alpha_i y_i = 0$, enforced by $A$ and $\mathbf{b}$, ensures the decision function is threshold-free, which is pivotal for the SVM model.

### Solving the QP Problem
The QP problem is solved using the `cvxopt` library's `qp` function, which finds the optimal set of Lagrange multipliers $\mathbf{\alpha}$. These multipliers are instrumental in constructing the model's decision function.


```python
cvxopt.solvers.options['show_progress'] = False
sol = cvxopt.solvers.qp(P, q, G, h, A, b)
```



## Prediction Process

Class prediction for new inputs is made by computing the dot product of the input with the weight vector and adding the bias. The class is determined by the resulting sign.

## The Static Method at Kernel Method

Using @staticmethod in the Kernel class for defining kernel functions like linear, polynomial, and Gaussian is a design choice that serves a specific purpose: emphasizes organization, ease of use, and the stateless nature of kernel functions, making them more accessible and logically grouped for users and developers working with the SVM implementation.

1. No Instance Required: Kernel functions do not need to access or modify the state of a class instance. They perform computations based solely on their input arguments. Therefore, there's no need for an instance of the Kernel class to use these methods.

2. Functionality Organization: It groups related functionality — in this case, different types of kernel functions — under a single class namespace. This makes the code more organized and intuitive. Users and developers can easily find all kernel-related functions within the Kernel class without needing to create an instance.

3. Ease of Use: Static methods can be called directly on the class without instantiation. This is particularly useful for the SVM class, which needs to utilize different kernels interchangeably. It can directly call Kernel.linear, Kernel.polynomial, or Kernel.gaussian without worrying about object creation, making the code cleaner and more efficient.

4. Extensibility and Maintenance: Should you need to add more kernel functions in the future, they can be added as static methods to the Kernel class. This centralizes kernel functions, making the codebase easier to maintain and extend.

5. Performance: While not the primary reason for using static methods in this context, avoiding the creation of class instances when they're not needed can lead to minor performance benefits, especially in scenarios where kernel functions are called repeatedly (such as in the training phase of an SVM).

# Concise Steps to Create a Virtual Environment Using venv in Python:

1. Open the Terminal or Command Prompt: On Windows, you can search for "cmd" or "Command Prompt". On macOS or Linux, open the "Terminal".

2. Navigate to the Project Directory: Use the `cd` command to navigate to the directory where you want to create the virtual environment. For example, `cd path/to/your/project`.

3. Create the Virtual Environment: Run the command to create the virtual environment within the project directory:

<python3 -m venv nome_do_ambiente>


On Windows, you might need to use `python` instead of `python3`. `environment_name` is the name you wish to give your virtual environment. This will create a new directory with that name in your project, containing the virtual environment.

4. Activate the Virtual Environment:

   - On Windows, execute:

     ```
     environment_name\Scripts\activate
     ```

   - On macOS or Linux, use:

     ```
     source environment_name/bin/activate
     ```

After activation, the name of your virtual environment will appear in the terminal prompt, indicating that any Python package you install using pip will be isolated in this environment.

5. Deactivate the Virtual Environment: When you are finished working in the virtual environment, you can deactivate it by running:

<deactivate>


6. Installing Packages
With the virtual environment activated, install the necessary packages using pip. The following command installs all packages at once:

<pip install tsfresh xgboost scikit-learn pandas matplotlib plotly seaborn numpy>


7. Generating the requirements.txt
After installing all the necessary packages, you can create the `requirements.txt` file to document the exact versions that the project requires. This is done with the following command:

```pip freeze > requirements.txt```


This file now contains all the dependencies of your Python project and their respective versions, making it easier to replicate the environment on other machines.

8. Using the requirements.txt
To install the dependencies in another environment using the `requirements.txt` you generated, use:

<pip install -r requirements.txt>


This process ensures that the same set of libraries and versions are installed, maintaining the consistency of the development environment.
