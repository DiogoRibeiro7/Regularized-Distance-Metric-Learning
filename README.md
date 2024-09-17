# Regularized Distance Metric Learning: Theory and Algorithm

## Overview

This repository contains the implementation of the algorithm presented in the paper "Regularized Distance Metric Learning: Theory and Algorithm." The paper introduces a method for learning a distance metric with a regularization framework that is robust to high-dimensional data. This approach has been shown to be effective in tasks like data classification and face recognition.

## Paper Summary

The key contributions of the paper are:
1. **Generalization Error Analysis**: The paper examines the generalization error of regularized distance metric learning. It shows that, with appropriate constraints, this error can be made independent of the data's dimensionality, making the method suitable for high-dimensional datasets.
2. **Online Learning Algorithm**: An efficient online learning algorithm for regularized distance metric learning is proposed, which is shown to have a provable regret bound.
3. **Empirical Evaluation**: The algorithm is empirically validated on classification tasks and face recognition, demonstrating its effectiveness and efficiency compared to state-of-the-art methods.

## Implementation Details

The implementation will cover the following components based on the paper:

### 1. Regularized Distance Metric Learning
The objective function for regularized distance metric learning is formulated as:

$$
\min_A \left\{ \frac{1}{2} \|A\|_F^2 + \frac{2C}{n(n-1)} \sum_{i<j} g \left( y_{i,j} \left[ 1 - \|x_i - x_j\|_A^2 \right] \right) : A \succeq 0, \text{tr}(A) \leq \eta(d) \right\}
$$

Where:
- \( A \) is the distance metric to be learned.
- \( g(\cdot) \) is a loss function, which is typically convex and Lipschitz continuous.
- \( \|A\|_F^2 \) is the Frobenius norm of the matrix \( A \).

### 2. Generalization Error Bound
The paper derives a generalization error bound that shows the stability of the algorithm with respect to changes in the training data. This provides a theoretical guarantee on the performance of the learned metric.

### 3. Online Learning Algorithm
The paper presents an online learning algorithm for distance metric learning, using a gradient-based update rule:

1. Receive a pair of training examples.
2. Compute the class label \( y_t \).
3. Update the metric \( A \) based on the hinge loss if the classification condition is violated.
4. Project the updated matrix back to the semidefinite cone.

The algorithm is computationally efficient and has a provable regret bound.

## Requirements

- Python 3.x
- NumPy
- SciPy

Additional libraries may be required for specific use cases like face recognition.

## Usage

### Data Preparation

The implementation will expect labeled training data of the form:
- \( \mathbf{X} = \{x_i\}_{i=1}^n \), where \( x_i \) are feature vectors.
- \( \mathbf{Y} = \{y_i\}_{i=1}^n \), where \( y_i \) are the corresponding class labels.

### Training the Metric

The training process involves:
1. Initializing the distance metric \( A \).
2. Iteratively updating \( A \) using the online learning algorithm.
3. Using the learned metric for downstream tasks like classification.

### Evaluation

The learned metric can be evaluated on tasks such as:
- **Data Classification**: Using \( k \)-Nearest Neighbors (k-NN) classifier with the learned metric.
- **Face Recognition**: Applying the learned metric to a face dataset.

## Example

A detailed example script will be provided in the `examples/` directory, demonstrating how to train the distance metric on a sample dataset and use it for classification.

## References

- Rong Jin, Shijun Wang, Yang Zhou. "Regularized Distance Metric Learning: Theory and Algorithm." (2009)


