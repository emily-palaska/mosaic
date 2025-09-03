# Random Code Synthesis
Query `Plot cross-decomposition results for two datasets.`
## Script Variables
- X_test:<br>
>X_test is a test dataset that is used to evaluate the performance of the model on unseen data.
- _:<br>
>The variable _ is a tuple containing two elements. The first element is a string that represents the name
- PCA:<br>
>PCA is a dimensionality reduction technique that is used to reduce the number of features in a dataset.
- X_test_kernel_pca:<br>
>X_test_kernel_pca is a numpy array containing the transformed testing data using KernelPCA. It is
- y_test:<br>
>The variable y_test is used to test the accuracy of the model. It is a list of labels
- fig:<br>
>fig is a figure object that is used to create multiple subplots. It is used to create a
- orig_data_ax:<br>
>It is a variable that is used to plot the original test data in the script. It is a
- plt:<br>
>Plotting library in Python.
- KernelPCA:<br>
>KernelPCA is a dimensionality reduction technique that uses a kernel trick to map the input data to a
- pca_proj_ax:<br>
>pca_proj_ax is a scatter plot of the testing data projected onto the first two principal components.
- np:<br>
>The variable np is a python library that provides a large number of mathematical functions and data structures. It
- X_train:<br>
>X_train is a pandas dataframe containing the features of the dataset. It is a matrix of shape (
- y_train:<br>
>y_train is a numpy array of size (n_samples, 1) containing the target values of
- clf_selected:<br>
>The clf_selected variable is a pipeline that is used to perform feature selection and scaling on the input data
- make_pipeline:<br>
>The variable make_pipeline is used to create a pipeline of machine learning algorithms. It takes a list of
- MinMaxScaler:<br>
>MinMaxScaler is a class that is used to scale the features to a given range. It is used
- f_classif:<br>
>f_classif is a function that is used to perform feature selection based on the F-statistic.
- svm_weights_selected:<br>
>svm_weights_selected is a variable that contains the weights of the selected features after applying the MinMaxScaler
- SelectKBest:<br>
>SelectKBest is a class that is used to select the most important features from a dataset. It
- print:<br>
>It is a function that prints the value of the variable. In this case, it prints the accuracy
- LinearSVC:<br>
>The LinearSVC class is a classifier that implements the linear SVM algorithm. It is a supervised learning
- var:<br>
>var is a variable that is used to multiply the value of comp by the value of var.
- comp:<br>
>The variable comp is a list of tuples, where each tuple represents a component of the principal components of
- search_95p:<br>
>It is a variable that represents the 95th percentile of the values of the variable search_05
- search_05p:<br>
>It is a variable that represents the probability that a randomly selected point from the training set belongs to the
- coverage_fraction:<br>
>The coverage_fraction variable is used to calculate the coverage of the prediction of the model. It is calculated
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: On the test set, the estimated confidence interval is slightly too narrow. Note, however, that we would need to wrap those metrics in a
cross-validation loop to assess their variability under data resampling.   Tuning the hyper-parameters of the quantile regressors  In the plot above,
we observed that the 5th percentile regressor seems to underfit and could not adapt to sinusoidal shape of the signal.  The hyper-parameters of the
model were approximately hand-tuned for the median regressor and there is no reason that the same hyper-parameters are suitable for the 5th percentile
regressor.  To confirm this hypothesis, we tune the hyper-parameters of a new regressor of the 5th percentile by selecting the best model parameters
by cross-validation on the pinball loss with alpha=0.05:   COMMENT: noqa: F401
```python
from sklearn.experimental import enable_halving_search_cv
```

### notebooks/dataset2/developing_estimators/sklearn_is_fitted.ipynb
CONTEXT:   `__sklearn_is_fitted__` as Developer API  The `__sklearn_is_fitted__` method is a convention used in scikit-learn for checking whether an
estimator object has been fitted or not. This method is typically implemented in custom estimator classes that are built on top of scikit-learn's base
classes like `BaseEstimator` or its subclasses.  Developers should use :func:`~sklearn.utils.validation.check_is_fitted` at the beginning of all
methods except `fit`. If they need to customize or speed-up the check, they can implement the `__sklearn_is_fitted__` method as shown below.  In this
example the custom estimator showcases the usage of the `__sklearn_is_fitted__` method and the `check_is_fitted` utility function as developer APIs.
The `__sklearn_is_fitted__` method checks fitted status by verifying the presence of the `_is_fitted` attribute.  An example custom estimator
implementing a simple classifier This code snippet defines a custom estimator class called `CustomEstimator` that extends both the `BaseEstimator` and
`ClassifierMixin` classes from scikit-learn and showcases the usage of the `__sklearn_is_fitted__` method and the `check_is_fitted` utility function.
COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:   Principal Component Regression vs Partial Least Squares Regression  This example compares [Principal Component
Regression](https://en.wikipedia.org/wiki/Principal_component_regression) (PCR) and [Partial Least Squares
Regression](https://en.wikipedia.org/wiki/Partial_least_squares_regression) (PLS) on a toy dataset. Our goal is to illustrate how PLS can outperform
PCR when the target is strongly correlated with some directions in the data that have a low variance.  PCR is a regressor composed of two steps:
first, :class:`~sklearn.decomposition.PCA` is applied to the training data, possibly performing dimensionality reduction; then, a regressor (e.g. a
linear regressor) is trained on the transformed samples. In :class:`~sklearn.decomposition.PCA`, the transformation is purely unsupervised, meaning
that no information about the targets is used. As a result, PCR may perform poorly in some datasets where the target is strongly correlated with
*directions* that have low variance. Indeed, the dimensionality reduction of PCA projects the data into a lower dimensional space where the variance
of the projected data is greedily maximized along each axis. Despite them having the most predictive power on the target, the directions with a lower
variance will be dropped, and the final regressor will not be able to leverage them.  PLS is both a transformer and a regressor, and it is quite
similar to PCR: it also applies a dimensionality reduction to the samples before applying a linear regressor to the transformed data. The main
difference with PCR is that the PLS transformation is supervised. Therefore, as we will see in this example, it does not suffer from the issue we just
mentioned.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
comp = comp * var
```

### notebooks/dataset2/clustering/plot_coin_segmentation.ipynb
CONTEXT: Compute and visualize the resulting regions   COMMENT: To view individual segments as appear comment in plt.pause(0.5)
```python
plt.show()
```

### notebooks/dataset2/decomposition/plot_kernel_pca.ipynb
CONTEXT: The samples from each class cannot be linearly separated: there is no straight line that can split the samples of the inner set from the
outer set.  Now, we will use PCA with and without a kernel to see what is the effect of using such a kernel. The kernel used here is a radial basis
function (RBF) kernel.   COMMENT:
```python
fig, (orig_data_ax, pca_proj_ax, KernelPCA) = plt.subplots(
    ncols=3, figsize=(14, 4)
)
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")

orig_data_ax.set_xlabel("Feature #0")

orig_data_ax.set_title("Testing data")
pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
pca_proj_ax.set_ylabel("Principal component #1")

pca_proj_ax.set_xlabel("Principal component #0")

pca_proj_ax.set_title("Projection of testing data\n using PCA")
KernelPCA.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
KernelPCA.set_ylabel("Principal component #1")

KernelPCA.set_xlabel("Principal component #0")

_ = KernelPCA.set_title("Projection of testing data\n using KernelPCA")
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: The plot looks qualitatively better than for the untuned models, especially for the shape of the of lower quantile.  We now quantitatively
evaluate the joint-calibration of the pair of estimators:   COMMENT:
```python
coverage_fraction(y_train, search_05p.predict(X_train), search_95p.predict(X_train))
```

### notebooks/dataset2/feature_selection/plot_feature_selection.ipynb
CONTEXT: In the total set of features, only the 4 of the original features are significant. We can see that they have the highest score with
univariate feature selection.   Compare with SVMs  Without univariate feature selection   COMMENT:
```python
clf_selected = make_pipeline(SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC())
clf_selected.fit(X_train, y_train)
print(
    "Classification accuracy after univariate feature selection: {:.3f}".format(
        clf_selected.score(X_test, y_test)
    )
)
svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()
```

## Code Concatenation
```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
comp = comp * var
plt.show()
fig, (orig_data_ax, pca_proj_ax, KernelPCA) = plt.subplots(
    ncols=3, figsize=(14, 4)
)
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")

orig_data_ax.set_xlabel("Feature #0")

orig_data_ax.set_title("Testing data")
pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
pca_proj_ax.set_ylabel("Principal component #1")

pca_proj_ax.set_xlabel("Principal component #0")

pca_proj_ax.set_title("Projection of testing data\n using PCA")
KernelPCA.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
KernelPCA.set_ylabel("Principal component #1")

KernelPCA.set_xlabel("Principal component #0")

_ = KernelPCA.set_title("Projection of testing data\n using KernelPCA")
coverage_fraction(y_train, search_05p.predict(X_train), search_95p.predict(X_train))
clf_selected = make_pipeline(SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC())
clf_selected.fit(X_train, y_train)
print(
    "Classification accuracy after univariate feature selection: {:.3f}".format(
        clf_selected.score(X_test, y_test)
    )
)
svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()
```
