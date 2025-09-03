# Reverse Embedding Code Synthesis
Query `Run GMM on sinusoidal synthetic data.`
## Script Variables
- time:<br>
>The variable time is a numpy array that contains 2000 values ranging from 0 to 8
- np:<br>
>The np is a Python package that provides a high-performance multidimensional array object, and tools for working
- s1:<br>
>The variable s1 is a sine function of time. It is used to generate a sine wave that
- hgbt:<br>
>hgbt is a HistGradientBoostingRegressor object which is used to perform gradient boosting regression.
- ax:<br>
>The variable ax is a matplotlib axis object that is used to plot the predicted and recorded average energy transfer
- y_train:<br>
>It is a numpy array containing the labels of the training data. The labels are integers from 0
- _:<br>
>The variable _ is a tuple containing the row and column indices of the subplot where the histogram will be
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses historical data to predict future values. It
- plt:<br>
>plt is a Python library that is used to create plots and graphs. It is a part of the
- X_train:<br>
>X_train is a 2D numpy array of shape (n_samples, n_features) containing the
- common_params:<br>
>It is a dictionary that contains the parameters that are common to all the trees in the ensemble. These
- max_iter:<br>
>The variable max_iter is a list of integers that represents the maximum number of iterations that the algorithm will
- S:<br>
>S is a matrix of size 2 x 2.
- A:<br>
>A is a matrix of size 3x3. It is a matrix of random numbers.
- X:<br>
>X is a matrix that represents the dot product of the matrix S and the transpose of the matrix A
- enumerate:<br>
>The enumerate() function returns an enumerate object. It is a sequence of tuples, where each tuple contains
- y_pred:<br>
>It is a variable that is used to predict the probability of a given class. It is a vector
- f1_score:<br>
>The f1_score function is used to calculate the F1 score of a given model. The F
- precision_score:<br>
>Precision_score is a metric used to measure the precision of a classifier. It is calculated as the ratio
- score_df:<br>
>The variable score_df is a dataframe that contains the scores of each classifier in the list clf_list.
- roc_auc_score:<br>
>The variable roc_auc_score is used to calculate the area under the receiver operating characteristic curve (ROC)
- pd:<br>
>It is a Python package that provides a high-performance, easy-to-use data structure for tabular data
- X_test:<br>
>X_test is a numpy array of shape (n_samples, n_features) containing the test data.
- scores:<br>
>Scores are a list of dictionaries containing the scores for each classifier. Each dictionary contains the scores for each
- name:<br>
>fig
- i:<br>
>The variable i is used to iterate over the list of classifiers. It is used to access the corresponding
- brier_score_loss:<br>
>Brier score loss is a measure of the accuracy of a binary classifier. It is calculated by taking
- metric:<br>
>The variable metric is a dictionary that stores the scores of each classifier on the test set. The scores
- y_prob:<br>
>y_prob is a variable that is used to store the probability of each class in the test data.
- y_test:<br>
>It is a test dataset that is used to evaluate the performance of the model. It is generated using
- clf_list:<br>
>It is a list of tuples, where each tuple contains a classifier and a name. The classifiers are
- log_loss:<br>
>The variable log_loss is a function that calculates the negative log likelihood of the predicted probabilities of the model
- clf:<br>
>clf is a classifier which is used to predict the class labels of the test data. It is a
- defaultdict:<br>
>The defaultdict is a dictionary that has a default value for each key. In this case, the default
- recall_score:<br>
>Recall score is a metric used to evaluate the performance of a classifier in predicting the positive class.
## Synthesis Blocks
### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Signal 1 : sinusoidal signal
```python
s1 = np.sin(2 * time)
```

### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Mix data
```python
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])

X = np.dot(S, A.T)

```

### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Mix data
```python
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])

X = np.dot(S, A.T)

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

### notebooks/dataset2/calibration/plot_calibration_curve.ipynb
CONTEXT: Notice that although calibration improves the `brier_score_loss` (a metric composed of calibration term and refinement term) and `log_loss`,
it does not significantly alter the prediction accuracy measures (precision, recall and F1 score). This is because calibration should not
significantly change prediction probabilities at the location of the decision threshold (at x = 0.5 on the graph). Calibration should however, make
the predicted probabilities more accurate and thus more useful for making allocation decisions under uncertainty. Further, ROC AUC, should not change
at all because calibration is a monotonic transformation. Indeed, no rank metrics are affected by calibration.   Linear support vector classifier
Next, we will compare:  * :class:`~sklearn.linear_model.LogisticRegression` (baseline) * Uncalibrated :class:`~sklearn.svm.LinearSVC`. Since SVC does
not output   probabilities by default, we naively scale the output of the   :term:`decision_function` into [0, 1] by applying min-max scaling. *
:class:`~sklearn.svm.LinearSVC` with isotonic and sigmoid   calibration (see `User Guide <calibration>`)   COMMENT: Add histogram
```python
scores = defaultdict(scores)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        scores = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[scores].append(metric(y_test, y_prob[:, 1]))
    for metric in [precision_score, recall_score, f1_score]:
        scores = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[scores].append(metric(y_test, y_pred))
    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df.round(decimals=3)
score_df
```

## Code Concatenation
```python
s1 = np.sin(2 * time)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])

X = np.dot(S, A.T)

A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])

X = np.dot(S, A.T)

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
scores = defaultdict(scores)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)
    for metric in [brier_score_loss, log_loss, roc_auc_score]:
        scores = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[scores].append(metric(y_test, y_prob[:, 1]))
    for metric in [precision_score, recall_score, f1_score]:
        scores = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[scores].append(metric(y_test, y_pred))
    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df.round(decimals=3)
score_df
```
