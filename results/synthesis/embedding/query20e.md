# Embedding Code Synthesis
Query `Use cross-validation to evaluate PCR.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to perform Principal Component Analysis on the data. It is
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- axes:<br>
>The variable axes in the above Python script are as follows
- X:<br>
>X is a pandas dataframe containing the features of the dataset. It is a 2D array containing
- rng:<br>
>The variable rng is a random number generator that is used to generate random numbers for the train-test split
- plt:<br>
>plt is a python library that is used for plotting graphs and charts. It is a popular library used
- fig:<br>
>fig is a variable that is used to store the figure object. It is used to plot the data
- y:<br>
>It is a numpy array containing the labels of the images in the dataset. The labels are integers between
- n_samples:<br>
>n_samples is the number of samples in the dataset. It is used to generate random noise in the
- scoring:<br>
>Scoring is a method used to evaluate the performance of a model. It is a way to measure
- hist_ordinal:<br>
>It is a variable that is used to represent the ordinal data type. It is a type of data
- one_hot_result:<br>
>The one_hot_result variable is a dictionary that contains the results of the cross-validation process for the one
- cross_validate:<br>
>It is a function that is used to evaluate the performance of a model on a dataset. It takes
- hist_dropped:<br>
>It is a variable that is used to store the results of the cross-validation process for the historical data
- ordinal_result:<br>
>OrdinalResult is a cross-validation result object that contains the results of the cross-validation process for the ordinal
- hist_native:<br>
>The variable hist_native is a HistGradientBoostingRegressor object that is used to fit a gradient boosting
- hist_one_hot:<br>
>It is a one hot encoding of the hist variable. This is a categorical variable that is used to
- n_cv_folds:<br>
>It is the number of folds used for cross validation. In this case, it is 3.
- native_result:<br>
>The variable native_result is a result of the cross_validate function which is used to evaluate the performance of
- dropped_result:<br>
>Dropped_result is a variable that is used to store the results of the cross-validation process for the
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_categorical.ipynb
CONTEXT:  Model comparison Finally, we evaluate the models using cross validation. Here we compare the models performance in terms of
:func:`~metrics.mean_absolute_percentage_error` and fit times.   COMMENT:
```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
scoring = "neg_mean_absolute_percentage_error"
n_cv_folds = 3
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  The data  We start by creating a simple dataset with two features. Before we even dive into PCR and PLS, we fit a PCA estimator to display
the two principal components of this dataset, i.e. the two directions that explain the most variance in the data.   COMMENT:
```python
y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first pca component", ylabel="y")
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second pca component", ylabel="y")
plt.tight_layout()
plt.show()
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
scoring = "neg_mean_absolute_percentage_error"
n_cv_folds = 3
dropped_result = cross_validate(hist_dropped, X, y, cv=n_cv_folds, scoring=scoring)
one_hot_result = cross_validate(hist_one_hot, X, y, cv=n_cv_folds, scoring=scoring)
ordinal_result = cross_validate(hist_ordinal, X, y, cv=n_cv_folds, scoring=scoring)
native_result = cross_validate(hist_native, X, y, cv=n_cv_folds, scoring=scoring)
pca = pcr.named_steps["pca"]
y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
axes[0].set(xlabel="Projected data onto first pca component", ylabel="y")
axes[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
axes[1].set(xlabel="Projected data onto second pca component", ylabel="y")
plt.tight_layout()
plt.show()
```
