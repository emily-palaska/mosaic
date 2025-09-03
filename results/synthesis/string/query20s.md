# String Code Synthesis
Query `Use cross-validation to evaluate PCR.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to perform Principal Component Analysis on the data. It is
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- X_test:<br>
>X_test is a matrix of 2 columns and 100 rows. It contains the values of the
- y_test:<br>
>The variable y_test is a vector of test data that is used to evaluate the performance of the model
- print:<br>
>The print function is used to display the output of the script to the console. It takes a single
- pls:<br>
>pls is a variable that is used to store the PLS regression model. It is used to predict
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
