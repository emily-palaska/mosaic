# Random Code Synthesis
Query `Analyze cluster stability across multiple runs.`
## Script Variables
- X_train:<br>
>X_train is a matrix of features used to train the model. It is a 2D array
- oa:<br>
>oa is a variable that is used to shrink the estimate of the OAS. It is used to
- loglik_oa:<br>
>The log likelihood of the observed data given the model parameters. It is a measure of how well the
- X_test:<br>
>X_test is a numpy array of size (500, 3) which contains the test data.
- OAS:<br>
>OAS is a class that is used to perform a logistic regression on the given data. It is
- hgbt:<br>
>hgbt is a HistGradientBoostingRegressor object which is used to perform gradient boosting regression.
- ax:<br>
>The variable ax is a matplotlib axis object that is used to plot the predicted and recorded average energy transfer
- y_train:<br>
>The variable y_train is a numpy array containing the target values for the training data. It is used
- _:<br>
>The variable _ is a placeholder for the number of iterations of the gradient boosting algorithm. It is used
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses historical data to predict future values. It
- plt:<br>
>The plt variable is a Python library that provides a graphical user interface for plotting data. It is a
- common_params:<br>
>It is a dictionary that contains the parameters that are common to all the trees in the ensemble. These
- max_iter:<br>
>The variable max_iter is a list of integers that represents the maximum number of iterations that the algorithm will
- mean_pinball_loss:<br>
>It is a function that calculates the mean pinball loss of a given model. The pinball loss
- pd:<br>
>It is a library that provides data structures and operations for manipulating and analyzing data in Python. It is
- sorted:<br>
>The variable sorted is a function that takes a list of tuples as input and returns a list of tuples
- y_pred:<br>
>It is a prediction of the model based on the training data. The model uses the training data to
- highlight_min:<br>
>highlight_min is a function that takes a dataframe as input and returns a boolean array. It is used
- all_models:<br>
>It is a dictionary that contains the Gradient Boosting Regressor models with different alpha values.
- name:<br>
>results = []
- mean_squared_error:<br>
>Mean squared error is a measure of the average of the squares of the differences between the values predicted by
- alpha:<br>
>Alpha is the quantile of the loss function used in Gradient Boosting Regression. It is a value
- results:<br>
>The variable results is a list of dictionaries. Each dictionary contains the metrics for a particular model. The
- metrics:<br>
>The variable metrics are used to evaluate the performance of the model in predicting the target variable. The metrics
- pca_2:<br>
>The variable pca_2 is a pipeline object that contains a PCA and a LinearRegression. The PCA
- y_test:<br>
>It is a test dataset that is used to evaluate the performance of the model. It is a set
- PCA:<br>
>PCA stands for Principal Component Analysis. It is a dimensionality reduction technique that transforms a set of correlated
- print:<br>
>The variable print is used to display the accuracy of the model on the test set. It is a
- make_pipeline:<br>
>Make_pipeline is a function that takes a list of estimators and returns a pipeline object. The pipeline
- LinearRegression:<br>
>The LinearRegression class is a linear regression model that fits a linear model with coefficients w to minimize the
- random_state:<br>
>It is a random number generator which is used to generate a sequence of random numbers. It is used
- np:<br>
>np is a library in python that provides a large collection of mathematical functions and data structures. It is
- clf:<br>
>clf is a Random Forest Classifier. It is a supervised machine learning algorithm that uses a decision tree to
- params:<br>
>params is a dictionary containing the parameters for the Gradient Boosting Classifier.
- x:<br>
>The variable x is a list of integers that represents the RGB values of the colors in the image.
- acc:<br>
>The variable acc is the accuracy of the Gradient Boosting Classifier model. It is calculated by comparing the
- n_estimators:<br>
>n_estimators is a parameter that controls the number of decision trees in the ensemble. It is a positive
- ensemble:<br>
>The ensemble variable is a dictionary that contains the parameters of the gradient boosting classifier. These parameters include the
- RandomForestClassifier:<br>
>RandomForestClassifier is a machine learning algorithm that uses a decision tree ensemble to classify data. It is
- X_train_valid:<br>
>X_train_valid is a numpy array containing 1000 observations of 2 features each. The observations
- plot:<br>
>Plot is a variable that is used to plot the data points in the given dataset. It is used
- enumerate:<br>
>enumerate() is a built-in function in Python that returns an enumerate object. It is used to create
- X:<br>
>X is a 2D array of size 1000x2. It represents the data points
- hdb:<br>
>It is a HDBSCAN() object. HDBSCAN() is a clustering algorithm that uses a
- scale:<br>
>The variable scale is used to control the sensitivity of the HDBSCAN algorithm to the noise in the
- axes:<br>
>The variable axes is a tuple of three elements. The first element is the figure object, which is
- HDBSCAN:<br>
>HDBSCAN is a clustering algorithm that uses a hierarchical density-based approach to cluster data. It is
- fig:<br>
>The variable fig is a figure object that is created using the plt.subplots() function. The function takes
- idx:<br>
>The variable idx is used to iterate over the different scales of the data. It is used to create
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_calibration_multiclass.ipynb
CONTEXT:  Fitting and calibration  First, we will train a :class:`~sklearn.ensemble.RandomForestClassifier` with 25 base estimators (trees) on the
concatenated train and validation data (1000 samples). This is the uncalibrated classifier.   COMMENT:
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_pred)
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: On the test set, the estimated confidence interval is slightly too narrow. Note, however, that we would need to wrap those metrics in a
cross-validation loop to assess their variability under data resampling.   Tuning the hyper-parameters of the quantile regressors  In the plot above,
we observed that the 5th percentile regressor seems to underfit and could not adapt to sinusoidal shape of the signal.  The hyper-parameters of the
model were approximately hand-tuned for the median regressor and there is no reason that the same hyper-parameters are suitable for the 5th percentile
regressor.  To confirm this hypothesis, we tune the hyper-parameters of a new regressor of the 5th percentile by selecting the best model parameters
by cross-validation on the pinball loss with alpha=0.05:   COMMENT: maximize the negative loss
```python
    greater_is_better=False,
```

### notebooks/dataset2/covariance_estimation/plot_covariance_estimation.ipynb
CONTEXT:  Compare different approaches to setting the regularization parameter  Here we compare 3 approaches:  * Setting the parameter by cross-
validating the likelihood on three folds   according to a grid of potential shrinkage parameters.  * A close formula proposed by Ledoit and Wolf to
compute   the asymptotically optimal regularization parameter (minimizing a MSE   criterion), yielding the :class:`~sklearn.covariance.LedoitWolf`
covariance estimate.  * An improvement of the Ledoit-Wolf shrinkage, the   :class:`~sklearn.covariance.OAS`, proposed by Chen et al. Its   convergence
is significantly better under the assumption that the data   are Gaussian, in particular for small samples.   COMMENT: Ledoit-Wolf optimal shrinkage
coefficient estimate
```python
oa = OAS()
loglik_oa = oa.fit(X_train).score(X_test)
```

### notebooks/dataset2/ensemble_methods/plot_hgbt_regression.ipynb
CONTEXT: With just a few iterations, HGBT models can achieve convergence (see
`sphx_glr_auto_examples_ensemble_plot_forest_hist_grad_boosting_comparison.py`), meaning that adding more trees does not improve the model anymore. In
the figure above, 5 iterations are not enough to get good predictions. With 50 iterations, we are already able to do a good job.  Setting `max_iter`
too high might degrade the prediction quality and cost a lot of avoidable computing resources. Therefore, the HGBT implementation in scikit-learn
provides an automatic **early stopping** strategy. With it, the model uses a fraction of the training data as internal validation set
(`validation_fraction`) and stops training if the validation score does not improve (or degrades) after `n_iter_no_change` iterations up to a certain
tolerance (`tol`).  Notice that there is a trade-off between `learning_rate` and `max_iter`: Generally, smaller learning rates are preferable but
require more iterations to converge to the minimum loss, while larger learning rates converge faster (less iterations/trees needed) but at the cost of
a larger minimum loss.  Because of this high correlation between the learning rate the number of iterations, a good practice is to tune the learning
rate along with all (important) other hyperparameters, fit the HBGT on the training set with a large enough value for `max_iter` and determine the
best `max_iter` via early stopping and some explicit `validation_fraction`.   COMMENT:
```python
common_params = {
    "max_iter": 1_000,
    "learning_rate": 0.3,
    "validation_fraction": 0.2,
    "random_state": 42,
    "categorical_features": None,
    "scoring": "neg_root_mean_squared_error",
}
hgbt = HistGradientBoostingRegressor(early_stopping=True, **common_params)
hgbt.fit(X_train, y_train)
_, ax = plt.subplots()
plt.plot(-hgbt.validation_score_)
_ = ax.set(
    xlabel="number of iterations",
    ylabel="root mean squared error",
    title=f"Loss of hgbt with early stopping (n_iter={hgbt.n_iter_})",
)
```

### notebooks/dataset2/clustering/plot_hdbscan.ipynb
CONTEXT: While standardizing data (e.g. using :class:`sklearn.preprocessing.StandardScaler`) helps mitigate this problem, great care must be taken to
select the appropriate value for `eps`.  HDBSCAN is much more robust in this sense: HDBSCAN can be seen as clustering over all possible values of
`eps` and extracting the best clusters from all possible clusters (see `User Guide <HDBSCAN>`). One immediate advantage is that HDBSCAN is scale-
invariant.   COMMENT:
```python
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
hdb = HDBSCAN()
for idx, scale in enumerate([1, 0.5, 3]):
    hdb.fit(X * scale)
    plot(
        X * scale,
        hdb.labels_,
        hdb.probabilities_,
        ax=axes[idx],
        parameters={"scale": scale},
    )
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_oob.ipynb
CONTEXT:   Gradient Boosting Out-of-Bag estimates Out-of-bag (OOB) estimates can be a useful heuristic to estimate the "optimal" number of boosting
iterations. OOB estimates are almost identical to cross-validation estimates but they can be computed on-the-fly without the need for repeated model
fitting. OOB estimates are only available for Stochastic Gradient Boosting (i.e. ``subsample < 1.0``), the estimates are derived from the improvement
in loss based on the examples not included in the bootstrap sample (the so-called out-of-bag examples). The OOB estimator is a pessimistic estimator
of the true test loss, but remains a fairly good approximation for a small number of trees. The figure shows the cumulative sum of the negative OOB
improvements as a function of the boosting iteration. As you can see, it tracks the test loss for the first hundred iterations but then diverges in a
pessimistic way. The figure also shows the performance of 3-fold cross validation which usually gives a better estimate of the test loss but is
computationally more demanding.  COMMENT: Fit classifier with out-of-bag estimates
```python
params = {
    "n_estimators": 1200,
    "max_depth": 3,
    "subsample": 0.5,
    "learning_rate": 0.01,
    "min_samples_leaf": 1,
    "random_state": 3,
}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy: {:.4f}".format(acc))
n_estimators = params["n_estimators"]
x = np.arange(n_estimators) + 1
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: Comparing the predicted median with the predicted mean, we note that the median is on average below the mean as the noise is skewed towards
high values (large outliers). The median estimate also seems to be smoother because of its natural robustness to outliers.  Also observe that the
inductive bias of gradient boosting trees is unfortunately preventing our 0.05 quantile to fully capture the sinoisoidal shape of the signal, in
particular around x=8. Tuning hyper-parameters can reduce this effect as shown in the last part of this notebook.   Analysis of the error metrics
Measure the models with :func:`~sklearn.metrics.mean_squared_error` and :func:`~sklearn.metrics.mean_pinball_loss` metrics on the training dataset.
COMMENT:
```python
results = []
for name, y_pred in sorted(all_models.items()):
    metrics = {"model": name}
    y_pred = y_pred.predict(X_train)
    for alpha in [0.05, 0.5, 0.95]:
        metrics["pbl=%1.2f" % alpha] = mean_pinball_loss(y_train, y_pred, alpha=alpha)
    metrics["MSE"] = mean_squared_error(y_train, y_pred)
    results.append(metrics)
pd.DataFrame(results).set_index("model").style.apply(highlight_min)
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: As expected, the unsupervised PCA transformation of PCR has dropped the second component, i.e. the direction with the lowest variance,
despite it being the most predictive direction. This is because PCA is a completely unsupervised transformation, and results in the projected data
having a low predictive power on the target.  On the other hand, the PLS regressor manages to capture the effect of the direction with the lowest
variance, thanks to its use of target information during the transformation: it can recognize that this direction is actually the most predictive. We
note that the first PLS component is negatively correlated with the target, which comes from the fact that the signs of eigenvectors are arbitrary.
We also print the R-squared scores of both estimators, which further confirms that PLS is a better alternative than PCR in this case. A negative
R-squared indicates that PCR performs worse than a regressor that would simply predict the mean of the target.   COMMENT:
```python
pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
pca_2.fit(X_train, y_train)
print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")
```

## Code Concatenation
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_pred)
    greater_is_better=False,
oa = OAS()
loglik_oa = oa.fit(X_train).score(X_test)
common_params = {
    "max_iter": 1_000,
    "learning_rate": 0.3,
    "validation_fraction": 0.2,
    "random_state": 42,
    "categorical_features": None,
    "scoring": "neg_root_mean_squared_error",
}
hgbt = HistGradientBoostingRegressor(early_stopping=True, **common_params)
hgbt.fit(X_train, y_train)
_, ax = plt.subplots()
plt.plot(-hgbt.validation_score_)
_ = ax.set(
    xlabel="number of iterations",
    ylabel="root mean squared error",
    title=f"Loss of hgbt with early stopping (n_iter={hgbt.n_iter_})",
)
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
hdb = HDBSCAN()
for idx, scale in enumerate([1, 0.5, 3]):
    hdb.fit(X * scale)
    plot(
        X * scale,
        hdb.labels_,
        hdb.probabilities_,
        ax=axes[idx],
        parameters={"scale": scale},
    )
params = {
    "n_estimators": 1200,
    "max_depth": 3,
    "subsample": 0.5,
    "learning_rate": 0.01,
    "min_samples_leaf": 1,
    "random_state": 3,
}
clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy: {:.4f}".format(acc))
n_estimators = params["n_estimators"]
x = np.arange(n_estimators) + 1
results = []
for name, y_pred in sorted(all_models.items()):
    metrics = {"model": name}
    y_pred = y_pred.predict(X_train)
    for alpha in [0.05, 0.5, 0.95]:
        metrics["pbl=%1.2f" % alpha] = mean_pinball_loss(y_train, y_pred, alpha=alpha)
    metrics["MSE"] = mean_squared_error(y_train, y_pred)
    results.append(metrics)
pd.DataFrame(results).set_index("model").style.apply(highlight_min)
pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
pca_2.fit(X_train, y_train)
print(f"PCR r-squared with 2 components {pca_2.score(X_test, y_test):.3f}")
```
