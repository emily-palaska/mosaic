# Random Code Synthesis
Query `Compare classifier performance using test accuracy.`
## Script Variables
- np:<br>
>The variable np is a python library that provides a large number of mathematical functions and data structures. It
- X_train:<br>
>X_train is a numpy array of shape (n_samples, n_features) containing the training data.
- y_train:<br>
>It is a variable that contains the training data for the decision tree classifier. It is used to train
- clf_selected:<br>
>The clf_selected variable is a pipeline that is used to perform feature selection and scaling on the input data
- X_test:<br>
>X_test is a matrix of test data. It is used to test the accuracy of the classifier.
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
- y_test:<br>
>y_test is a variable that contains the test data. It is used to evaluate the performance of the
- LinearSVC:<br>
>The LinearSVC class is a classifier that implements the linear SVM algorithm. It is a supervised learning
- y:<br>
>The variable y is the predicted probability of the data point being a 1 (positive) or
- X_transformed:<br>
>X_transformed is a matrix of 1000 rows and 100 columns. It contains the original
- nb:<br>
>The variable nb is a BernoulliNB() object. It is a type of classification algorithm that
- BernoulliNB:<br>
>BernoulliNB is a class in scikit-learn which is used to fit a Bernou
- hgbt:<br>
>hgbt is a HistGradientBoostingRegressor object which is used to perform gradient boosting regression.
- ax:<br>
>ax is a variable that is used to plot the number of nodes and depth of the trees as a
- _:<br>
>The variable _ is a placeholder for the number of iterations of the gradient boosting algorithm. It is used
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses historical data to predict future values. It
- plt:<br>
>plt is a python library that is used for plotting data in a variety of ways. It is a
- common_params:<br>
>It is a dictionary that contains the parameters that are common to all the trees in the ensemble. These
- max_iter:<br>
>The variable max_iter is a list of integers that represents the maximum number of iterations that the algorithm will
- ccp_alphas:<br>
>ccp_alphas is a variable that stores the alpha values for the linear support vector machine. It
- fig:<br>
>fig is a figure object which is used to plot the graph. It is a container for the axes
- clfs:<br>
>clfs is a list of decision trees. It contains the trees that are used to predict the outcome
- test_scores:<br>
>The variable test_scores is a list of scores that represent the accuracy of the model on the testing set
- train_scores:<br>
>train_scores is a list of scores that are obtained from the training data set. The scores are calculated
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_random_forest_embedding.ipynb
CONTEXT:   Hashing feature transformation using Totally Random Trees  RandomTreesEmbedding provides a way to map data to a very high-dimensional,
sparse representation, which might be beneficial for classification. The mapping is completely unsupervised and very efficient.  This example
visualizes the partitions given by several trees and shows how the transformation can also be used for non-linear dimensionality reduction or non-
linear classification.  Points that are neighboring often share the same leaf of a tree and therefore share large parts of their hashed
representation. This allows to separate two concentric circles simply based on the principal components of the transformed data with truncated SVD.
In high-dimensional spaces, linear classifiers often achieve excellent accuracy. For sparse binary data, BernoulliNB is particularly well-suited. The
bottom row compares the decision boundary obtained by BernoulliNB in the transformed space with an ExtraTreesClassifier forests learned on the
original data.  COMMENT: Learn a Naive Bayes classifier on the transformed data
```python
nb = BernoulliNB()
nb.fit(X_transformed, y)
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

### notebooks/dataset2/decision_trees/plot_cost_complexity_pruning.ipynb
CONTEXT:  Accuracy vs alpha for training and testing sets When ``ccp_alpha`` is set to zero and keeping the other default parameters of
:class:`DecisionTreeClassifier`, the tree overfits, leading to a 100% training accuracy and 88% testing accuracy. As alpha increases, more of the tree
is pruned, thus creating a decision tree that generalizes better. In this example, setting ``ccp_alpha=0.015`` maximizes the testing accuracy.
COMMENT:
```python
train_scores = [clfs.score(X_train, y_train) for clfs in clfs]
test_scores = [clfs.score(X_test, y_test) for clfs in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
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
nb = BernoulliNB()
nb.fit(X_transformed, y)
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
train_scores = [clfs.score(X_train, y_train) for clfs in clfs]
test_scores = [clfs.score(X_test, y_test) for clfs in clfs]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
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
