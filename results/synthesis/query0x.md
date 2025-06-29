# Exhaustive Code Synthesis
Query `Create Regression, SVM, Tree, AdaBoost and Bayes classifiers. Compare the classifiers and plot them.`
## Script Variables
- X_train:<br>
>X_train is a matrix containing the first two features of the Iris dataset. It is used to train
- evaluation_results:<br>
>The evaluation_results variable is a pandas DataFrame that contains the results of the evaluation of the model. It
- n_classifiers:<br>
>n_classifiers is a variable that stores the number of classifiers used in the script. It is used
- classifier_idx:<br>
>The classifier_idx variable is used to iterate over the classifiers dictionary. It is used to access the name
- X_test:<br>
>X_test is a matrix of size (n_samples, n_features) where n_samples is the number
- classifiers:<br>
>The variable classifiers are used to determine the number of classifiers used in the model. This is done by
- iris:<br>
>iris is a dataset that contains information about 150 flowers of the Iris species. The dataset contains
- y_train:<br>
>y_train is a variable that contains the target values for the training data. It is used to evaluate
- mpl:<br>
>mpl is a python package that provides a comprehensive set of tools for creating and manipulating plots in a variety
- fig:<br>
>fig is a variable that is used to create a figure object. It is used to create a plot
- levels:<br>
>levels
- roc_auc_test:<br>
>The roc_auc_test is the area under the curve of the receiver operating characteristic (ROC) curve.
- name:<br>
>The variable name is "evaluation_results". It is a list of dictionaries that contain the accuracy, roc
- classifier:<br>
>The variable classifier is a machine learning algorithm that is used to identify the most important features in a dataset
- y_test:<br>
>y_test is a variable that is used to test the model. It is a list of integers that
- plt:<br>
>plt is a module that provides a large suite of command line tools for creating plots. It is a
- accuracy_score:<br>
>Accuracy score is a measure of the quality of a binary classification model. It is calculated as the ratio
- axes:<br>
>The variable axes are used to display the decision boundary of the classifier on the training data. The classifier
- len:<br>
>len is a function that returns the length of an object. In this case, it is used to
- accuracy_test:<br>
>Accuracy test is a metric used to evaluate the performance of a machine learning model in classifying data.
- y_pred:<br>
>y_pred is a variable that stores the predicted values of the test data set. It is used to
- enumerate:<br>
>enumerate() is a built-in function that returns an iterator over the indices and values of a sequence.
- log_loss_test:<br>
>The log_loss_test variable is a measure of the accuracy of the model in predicting the class labels of
- log_loss:<br>
>The log_loss function is used to calculate the loss function for a binary classification problem. It is a
- roc_auc_score:<br>
>The roc_auc_score is a function that calculates the area under the receiver operating characteristic curve (ROC)
- y_pred_proba:<br>
>It is a probability vector of length 3, which represents the probability of each class (0,
- make_pipeline:<br>
>make_pipeline() is a function in scikit-learn that allows us to create a pipeline of machine
- SplineTransformer:<br>
>SplineTransformer is a class that transforms the input data into a new feature space using splines.
- LogisticRegression:<br>
>Logistic regression is a type of classification algorithm that is used to predict the probability of a given outcome
- RBF:<br>
>RBF is an acronym for Radial Basis Function. It is a type of kernel function used in
- Nystroem:<br>
>Nystroem is a kernel-based method for dimensionality reduction. It is a wrapper around a
- HistGradientBoostingClassifier:<br>
>HistGradientBoostingClassifier is a machine learning algorithm that uses a gradient boosting technique to fit a histogram
- KBinsDiscretizer:<br>
>KBinsDiscretizer is a class used to discretize continuous features into a fixed number of bins
- GaussianProcessClassifier:<br>
>The GaussianProcessClassifier is a classifier that uses a Gaussian process to make predictions. It is a non
- PolynomialFeatures:<br>
>PolynomialFeatures is a class that is used to create polynomial features from the input data. It is
## Synthesis Blocks
### notebooks/dataset2/classification/plot_classification_probability.ipynb
CONTEXT:  Probabilistic classifiers  We will plot the decision boundaries of several classifiers that have a `predict_proba` method. This will allow
us to visualize the uncertainty of the classifier in regions where it is not certain of its prediction.   COMMENT:
```python
classifiers = {
    "Logistic regression\n(C=0.01)": LogisticRegression(C=0.1),
    "Logistic regression\n(C=1)": LogisticRegression(C=100),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(),
    "Logistic regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}
```

### notebooks/dataset2/classification/plot_classification_probability.ipynb
CONTEXT:  Plotting the decision boundaries  For each classifier, we plot the per-class probabilities on the first three columns and the probabilities
of the most likely class on the last column.   COMMENT: Ensure legend not cut off
```python
mpl.rcParams["savefig.bbox"] = "tight"
fig, axes = plt.subplots(
    nrows=n_classifiers,
    ncols=len(iris.target_names) + 1,
    figsize=(4 * 2.2, n_classifiers * 2.2),
)
evaluation_results = []
levels = 100
for classifier_idx, (name, classifier) in enumerate(classifiers.items()):
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    log_loss_test = log_loss(y_test, y_pred_proba)
    evaluation_results.append(
        {
            "name": name.replace("\n", " "),
            "accuracy": accuracy_test,
            "roc_auc": roc_auc_test,
            "log_loss": log_loss_test,
        }
    )
```

## Code Concatenation
```python
classifiers = {
    "Logistic regression\n(C=0.01)": LogisticRegression(C=0.1),
    "Logistic regression\n(C=1)": LogisticRegression(C=100),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(),
    "Logistic regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}
mpl.rcParams["savefig.bbox"] = "tight"
fig, axes = plt.subplots(
    nrows=n_classifiers,
    ncols=len(iris.target_names) + 1,
    figsize=(4 * 2.2, n_classifiers * 2.2),
)
evaluation_results = []
levels = 100
for classifier_idx, (name, classifier) in enumerate(classifiers.items()):
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    log_loss_test = log_loss(y_test, y_pred_proba)
    evaluation_results.append(
        {
            "name": name.replace("\n", " "),
            "accuracy": accuracy_test,
            "roc_auc": roc_auc_test,
            "log_loss": log_loss_test,
        }
    )
```
