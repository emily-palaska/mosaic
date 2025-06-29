# Embedding Code Synthesis
Query `How to perform cross_decomposition`
## Script Variables
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- ax:<br>
>ax is a scatter plot object that is used to plot the training data points on the scatter plot.
- y_min:<br>
>The variable y_min is the minimum value of the y axis. It is used to set the limits
- x_min:<br>
>x_min is the minimum value of the first column of the dataset X. It is used to determine
- cm_bright:<br>
>cm_bright is a colormap that is used to color the scatter plot. It is a color map
- X_test:<br>
>X_test is a 2D array containing the test data. It is used to plot the scatter
- i:<br>
>The variable i is a counter that is used to keep track of the number of plots that have been
- y_test:<br>
>The variable y_test is a numpy array containing the true labels of the test data. It is used
- clf:<br>
>It is a classifier that is used to predict the class of a given data point.
- svm:<br>
>svm is a short form of support vector machine. It is a machine learning algorithm that is used for
- print:<br>
>The print function is used to display the output of a Python expression on the screen. It is a
- y_true:<br>
>It is a list of all the true labels in the dataset. It is used to calculate the accuracy
- metrics:<br>
>Confusion matrix
- len:<br>
>len is a built-in function that returns the length of an object. In this case, it is
- cm:<br>
>cm is a confusion matrix that is used to compare the actual values of the target variable with the predicted
- y_pred:<br>
>It is a confusion matrix which is used to compare the predicted values with the actual values. It is
- range:<br>
>The range of the variable is from 0 to 4. The variable is used to iterate through
- pred:<br>
>pred is a variable that is used to iterate through the confusion matrix. It is used to access the
- predicted:<br>
>The variable predicted is the predicted value of the image. It is used to determine the classification of the
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:   Compare cross decomposition methods  Simple usage of various cross decomposition algorithms:  - PLSCanonical - PLSRegression, with
multivariate response, a.k.a. PLS2 - PLSRegression, with univariate response, a.k.a. PLS1 - CCA  Given 2 multivariate covarying two-dimensional
datasets, X, and Y, PLS extracts the 'directions of covariance', i.e. the components of each datasets that explain the most shared variance between
both datasets. This is apparent on the **scatterplot matrix** display: components 1 in dataset X and dataset Y are maximally correlated (points lie
around the first diagonal). This is also true for components 2 in both dataset, however, the correlation across datasets for different components is
weak: the point cloud is very spherical.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import numpy as np
n = 500
```

### notebooks/dataset2/classification/plot_lda_qda.ipynb
CONTEXT: We generate three datasets. In the first dataset, the two classes share the same covariance matrix, and this covariance matrix has the
specificity of being spherical (isotropic). The second dataset is similar to the first one but does not enforce the covariance to be spherical.
Finally, the third dataset has a non-spherical covariance matrix for each class.   COMMENT:
```python
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT:  Classification  To apply a classifier on this data, we need to flatten the images, turning each 2-D array of grayscale values from shape
``(8, 8)`` into shape ``(64,)``. Subsequently, the entire dataset will be of shape ``(n_samples, n_features)``, where ``n_samples`` is the number of
images and ``n_features`` is the total number of pixels in each image.  We can then split the data into train and test subsets and fit a support
vector classifier on the train samples. The fitted classifier can subsequently be used to predict the value of the digit for the samples in the test
subset.   COMMENT: Create a classifier: a support vector classifier
```python
clf = svm.SVC(gamma=0.001)
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT: :func:`~sklearn.metrics.classification_report` builds a text report showing the main classification metrics.   COMMENT:
```python
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT: If the results from evaluating a classifier are stored in the form of a `confusion matrix <confusion_matrix>` and not in terms of `y_true`
and `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report` as follows:   COMMENT: For each cell in the confusion matrix, add
the corresponding ground truths and predictions to the lists
```python
for y_pred in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [y_pred] * cm[y_pred][pred]
        y_pred += [pred] * cm[y_pred][pred]
print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
```

### notebooks/dataset2/classification/plot_classifier_comparison.ipynb
CONTEXT:   Classifier comparison  A comparison of several classifiers in scikit-learn on synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these
examples does not necessarily carry over to real datasets.  Particularly in high-dimensional spaces, data can more easily be separated linearly and
the simplicity of classifiers such as naive Bayes and linear SVMs might lead to better generalization than is achieved by other classifiers.  The
plots show training points in solid colors and testing points semi-transparent. The lower right shows the classification accuracy on the test set.
COMMENT: Plot the testing points
```python
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
ax.set_xlim(x_min, y_min)
ax.set_ylim(y_min, y_min)
ax.set_xticks(())
ax.set_yticks(())
i += 1
```

## Code Concatenation
```python
import numpy as np
n = 500
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
clf = svm.SVC(gamma=0.001)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
for y_pred in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [y_pred] * cm[y_pred][pred]
        y_pred += [pred] * cm[y_pred][pred]
print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
ax.set_xlim(x_min, y_min)
ax.set_ylim(y_min, y_min)
ax.set_xticks(())
ax.set_yticks(())
i += 1
```
