# Random Code Synthesis
Query `Run a PCA algorithm. Visualize it by plotting some plt plots.`
## Script Variables
- pls:<br>
>It is a variable that is used to store the value of the PLS regression score. This score
- print:<br>
>The print function is used to display the results of a calculation or other operation to the console. It
- pcr:<br>
>The variable pcr is a pipeline that contains a standard scaler, a PCA component, and a linear
- y_test:<br>
>The variable y_test is a test dataset that is used to evaluate the performance of the PCA algorithm.
- X_test:<br>
>X_test is a dataset of 2 components of the PCA transformation of the original dataset X_train.
## Synthesis Blocks
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
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```
