# Exhaustive Code Synthesis
Query `Run a PCA algorithm. Visualize it by plotting some plt plots.`
## Script Variables
- print:<br>
>The variable print is used to print the correlation matrix of the input data. It is used to check
- q:<br>
>The variable q is the number of components used in the PLS regression model. It is used to
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- Y:<br>
>Y is a matrix of size (n, 4) where n is the number of samples.
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. PLS
- X:<br>
>X is a matrix of size n x q where n is the number of samples and q is the
- pls2:<br>
>pls2 is a PLSRegression object that is used to fit the data and predict the output.
- B:<br>
>B is a matrix of size (q, p) where q is the number of components and p
- X_train:<br>
>X_train is a matrix containing the first two features of the Iris dataset. It is used to train
- classifier_idx:<br>
>The classifier_idx variable is used to iterate over the classifiers dictionary. It is used to access the name
- label:<br>
>The variable label is a list of colors that are used to represent the different classes in the dataset.
- X_test:<br>
>X_test is a matrix of size (n_samples, n_features) where n_samples is the number
- levels:<br>
>levels
- max_class_disp:<br>
>max_class_disp is a variable that is used to store the maximum class dispersion of the surface. It
- name:<br>
>The variable name is "evaluation_results". It is a list of dictionaries that contain the accuracy, roc
- classifier:<br>
>The variable classifier is a machine learning algorithm that is used to identify the most important features in a dataset
- y_test:<br>
>y_test is a variable that is used to test the model. It is a list of integers that
- y_unique:<br>
>The variable y_unique is a list of unique values in the y-axis of the surface plot. It
- mask_label:<br>
>The variable mask_label is used to select the rows of the test dataset that correspond to the label of
- axes:<br>
>The variable axes are used to display the decision boundary of the classifier on the training data. The classifier
- len:<br>
>len is a function that returns the length of an object. In this case, it is used to
- scatter_kwargs:<br>
>It is a dictionary that contains the keyword arguments for the scatter() function. The keyword arguments are used
- DecisionBoundaryDisplay:<br>
>It is a class that displays the decision boundary of a classifier. It is used to visualize the decision
- evaluation_results:<br>
>The evaluation_results variable is a pandas DataFrame that contains the results of the evaluation of the model. It
- n_classifiers:<br>
>n_classifiers is a variable that stores the number of classifiers used in the script. It is used
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
- roc_auc_test:<br>
>The roc_auc_test is the area under the curve of the receiver operating characteristic (ROC) curve.
- plt:<br>
>plt is a module that provides a large suite of command line tools for creating plots. It is a
- accuracy_score:<br>
>Accuracy score is a measure of the quality of a binary classification model. It is calculated as the ratio
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
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with multivariate response, a.k.a. PLS2   COMMENT:
```python
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
```

### notebooks/dataset2/classification/plot_classification_probability.ipynb
CONTEXT:  Plotting the decision boundaries  For each classifier, we plot the per-class probabilities on the first three columns and the probabilities
of the most likely class on the last column.   COMMENT: add column that shows all classes by plotting class with max 'predict_proba'
```python
max_class_disp = DecisionBoundaryDisplay.from_estimator(
    classifier,
    X_train,
    response_method="predict_proba",
    class_of_interest=None,
    ax=axes[classifier_idx, len(y_unique)],
    vmin=0,
    vmax=1,
    levels=levels,
)
for label in y_unique:
    mask_label = y_test == label
    axes[classifier_idx, 3].scatter(
        X_test[mask_label, 0],
        X_test[mask_label, 1],
        c=max_class_disp.multiclass_colors_[[label], :],
        **scatter_kwargs,
    )
axes[classifier_idx, 3].set(xticks=(), yticks=())
axes[classifier_idx, 3].set_title("Max class")
axes[classifier_idx, 0].set_ylabel(name)
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
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
max_class_disp = DecisionBoundaryDisplay.from_estimator(
    classifier,
    X_train,
    response_method="predict_proba",
    class_of_interest=None,
    ax=axes[classifier_idx, len(y_unique)],
    vmin=0,
    vmax=1,
    levels=levels,
)
for label in y_unique:
    mask_label = y_test == label
    axes[classifier_idx, 3].scatter(
        X_test[mask_label, 0],
        X_test[mask_label, 1],
        c=max_class_disp.multiclass_colors_[[label], :],
        **scatter_kwargs,
    )
axes[classifier_idx, 3].set(xticks=(), yticks=())
axes[classifier_idx, 3].set_title("Max class")
axes[classifier_idx, 0].set_ylabel(name)
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
