# Reverse Embedding Code Synthesis
Query `Create classifiers with different types and names. Compare the classifiers and plot them.`
## Script Variables
- axes:<br>
>The axes variable is used to create a grid of axes in a figure. It is a list of
- name:<br>
>max_class_disp
- classifier_idx:<br>
>It is a variable that represents the classifier used to generate the decision boundary display. In this case,
- classifier:<br>
>The variable classifier is a machine learning algorithm that is used to classify data into different categories. It is
- fig:<br>
>fig is a variable that is used to store the figure object that is created by the script. It
- classifiers:<br>
>The variable classifiers are used to determine the number of classifiers to be used in the script. They are
- y_unique:<br>
>y_unique is a list of unique values in the y_test variable. It is used to create a
- plt:<br>
>plt is a Python library that provides a wide range of plotting functions and tools for creating and customizing
- X_train:<br>
>X_train is a numpy array of shape (n_samples, n_features) containing the training data.
- y_test:<br>
>y_test is the test set of Iris flower data. It contains the target values of the test set
- levels:<br>
>levels
- X_test:<br>
>The variable X_test is a test dataset that is used to evaluate the performance of the model. It
- y_train:<br>
>It is a target variable that contains the species of the iris flower. It is used to split the
- iris:<br>
>It is a dataset that contains information about the iris flowers. It has 3 classes of iris flowers
- len:<br>
>len is a variable that is used to count the number of elements in a list or tuple. It
- y_pred:<br>
>The variable y_pred is a prediction of the output of the model. It is used to determine the
- n_classifiers:<br>
>n_classifiers is the number of classifiers used in the script. It is used to determine the number
- evaluation_results:<br>
>It is a pandas DataFrame object that contains the evaluation results of the model. The columns of the DataFrame
## Synthesis Blocks
### notebooks/plot_classification_probability.ipynb
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
    for name in y_unique:
```

### notebooks/plot_lda_qda.ipynb
CONTEXT: We generate three datasets. In the first dataset, the two classes share the same covariance matrix, and this covariance matrix has the
specificity of being spherical (isotropic). The second dataset is similar to the first one but does not enforce the covariance to be spherical.
Finally, the third dataset has a non-spherical covariance matrix for each class.   COMMENT:
```python
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
```

## Code Concatenation
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
    for name in y_unique:
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
```
