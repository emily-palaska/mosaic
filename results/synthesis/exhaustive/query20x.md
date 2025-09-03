# Exhaustive Code Synthesis
Query `Use cross-validation to evaluate PCR.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to perform Principal Component Analysis on the data. It is
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- axes:<br>
>The variable axes in the above Python script are as follows
- X:<br>
>X is a numpy array containing the data points for the first PCA component. It is a 2
- PCA:<br>
>PCA stands for Principal Component Analysis. It is a dimensionality reduction technique that transforms a set of correlated
- rng:<br>
>The variable rng is a random number generator that is used to generate random numbers for the train-test split
- plt:<br>
>plt is a module in python which is used for plotting graphs. It is a part of the matplotlib
- fig:<br>
>fig is a variable that is used to store the figure object. It is used to plot the data
- y:<br>
>Variable y is the dependent variable in the script. It is used to predict the value of the dependent
- n_samples:<br>
>n_samples is the number of samples in the dataset. It is used to generate random noise in the
- TimeSeriesSplit:<br>
>TimeSeriesSplit is a class that splits a time series into a number of time series. The class
- ts_cv:<br>
>TimeSeriesSplit is a cross-validation object that splits the time series into a number of folds. The
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/dataset2/ensemble_methods/plot_hgbt_regression.ipynb
CONTEXT: Observe that `nswdemand` and `vicdemand` seem already monotonic without constraint. This is a good example to show that the model with
monotonicity constraints is "overconstraining".  Additionally, we can verify that the predictive quality of the model is not significantly degraded by
introducing the monotonic constraints. For such purpose we use :class:`~sklearn.model_selection.TimeSeriesSplit` cross-validation to estimate the
variance of the test score. By doing so we guarantee that the training data does not succeed the testing data, which is crucial when dealing with data
that have a temporal relationship.   COMMENT: a week has 336 samples
```python
ts_cv = TimeSeriesSplit(n_splits=5, gap=48, test_size=336)
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  The data  We start by creating a simple dataset with two features. Before we even dive into PCR and PLS, we fit a PCA estimator to display
the two principal components of this dataset, i.e. the two directions that explain the most variance in the data.   COMMENT:
```python
y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first PCA component", ylabel="y")
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second PCA component", ylabel="y")
plt.tight_layout()
plt.show()
```

## Code Concatenation
```python
pca = pcr.named_steps["pca"]
ts_cv = TimeSeriesSplit(n_splits=5, gap=48, test_size=336)
y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first PCA component", ylabel="y")
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second PCA component", ylabel="y")
plt.tight_layout()
plt.show()
```
