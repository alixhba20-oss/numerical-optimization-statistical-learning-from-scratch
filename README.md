# Numerical Optimization & Statistical Learning From Scratch

Implementation of first-order, second-order, and adaptive optimization algorithms alongside regularized logistic regression (MLE) in pure NumPy—without high-level ML wrappers.

## Mathematical Formulation
For binary classification with parameters $\theta = (b, w_1, \dots, w_d)$ and $\ell_2$ regularization $\lambda$:
$$\mathcal{L}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\right] + \frac{\lambda}{2}\|w\|_2^2, \qquad p_i = \sigma(x_i^\top \theta)$$
* **Batch Gradient Descent:** $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$
* **Newton–Raphson:** $\theta \leftarrow \theta - H(\theta)^{-1}\nabla_\theta \mathcal{L}(\theta)$ using the exact Hessian $H(\theta)$
* **Mini-Batch SGD & Adam:** Stochastic mini-batch updates with first/second moment bias correction

## Key Results (`01_Optimization_and_Logistic_MLE.ipynb`)
1. **Curvature vs. First-Order Convergence:** Newton–Raphson reaches near-exact convergence in far fewer iterations than batch gradient descent by exploiting second-order Hessian information.
2. **Adaptive vs. Stochastic Updates:** Compares mini-batch SGD against a from-scratch Adam optimizer across training epochs.
3. **Classification & Decision Boundaries:** Achieves 97% training accuracy and visualizes learned linear decision boundaries across feature space.

## Tools
Python, NumPy, Matplotlib, Google Colab
