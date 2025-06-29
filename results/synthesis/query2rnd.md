# Random Code Synthesis
Query `Graph operations`
## Script Variables
- print:<br>
>The variable print is used to print the correlation matrix of the input data. It is used to check
- X_test:<br>
>X_test is a matrix of size (n_samples, n_features) where n_samples is the number
- Y_test:<br>
>Y_test is a variable that is used to test the performance of the PLSCanonical algorithm on
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- X_train:<br>
>X_train is a dataset of 1000 observations and 20 variables. It is used to train
- l1:<br>
>l1 is a numpy array of size n which is used to generate the latent variables l1 and
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- Y_train:<br>
>Y_train is a matrix of size 1000x1, which contains the target values for the
- l2:<br>
>l2 is a random variable that is generated using the normal distribution with a size of n. It
- Y:<br>
>Y is a matrix of size (n, 4) where n is the number of samples.
- X:<br>
>X is a matrix of size n x q where n is the number of samples and q is the
- latents:<br>
>latents is a matrix of 4 rows and n columns. Each row represents a latent variable.
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Dataset based latent variables model   COMMENT:
```python
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)
latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))
X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))
```

## Code Concatenation
```python
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)
latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))
X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))
```
