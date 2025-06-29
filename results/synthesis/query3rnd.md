# Random Code Synthesis
Query `How to perform cross_decomposition`
## Script Variables
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- y:<br>
>The variable y is a 1000x1 matrix containing the actual values of the dependent variable.
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- p:<br>
>p is the number of components in the PLS model. In this case, it is 3
- pls1:<br>
>pls1 is a PLSRegression object which is used to perform PLS regression. PLS regression
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. PLS
- X:<br>
>X is a matrix of size n x q where n is the number of samples and q is the
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with multivariate response, a.k.a. PLS2   COMMENT: compare pls2.coef_ with B
```python
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(X, y)
```

## Code Concatenation
```python
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(X, y)
```
