# Exhaustive Code Synthesis
Query `PCA algorithm`
## Script Variables
- y_test:<br>
>y_test is a numpy array of size (1000,) which is the test data set for the
- print:<br>
>The variable print is used to display the results of the PLS regression model on the test data set
- pls:<br>
>It is a variable that is used to store the result of the PLS regression model. The P
- X_test:<br>
>X_test is a numpy array containing the test data. It is used to predict the values of y
- pcr:<br>
>The variable pcr is a function that calculates the r-squared value of a given dataset. It
- pca:<br>
>pca is a PCA object which is used to perform Principal Component Analysis. PCA is a dimensionality
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  Projection on one component and predictive power  We now create two regressors: PCR and PLS, and for our illustration purposes we set the
number of components to 1. Before feeding the data to the PCA step of PCR, we first standardize it, as recommended by good practice. The PLS estimator
has built-in scaling capabilities.  For both models, we plot the projected data onto the first component against the target. In both cases, this
projected data is what the regressors will use as training data.   COMMENT:
```python
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```

## Code Concatenation
```python
pca = pcr.named_steps["pca"]
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```
