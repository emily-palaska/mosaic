# Embedding Code Synthesis
Query `Create a regression model.`
## Script Variables
- RBF:<br>
>RBF is an acronym for Radial Basis Function. It is a kernel function that is used in
- PolynomialFeatures:<br>
>PolynomialFeatures is a class that takes in a dataset and transforms it into a new dataset with polynomial
- Nystroem:<br>
>Nystroem is a kernel-based method that is used to transform the input data into a high
- classifiers:<br>
>The variable classifiers are used to determine the number of classifiers to be used in the script. They are
- GaussianProcessClassifier:<br>
>The GaussianProcessClassifier is a machine learning classifier that uses Gaussian processes to make predictions. It is a
- LogisticRegression:<br>
>Logistic regression is a supervised machine learning algorithm that is used for classification problems. It is a type
- HistGradientBoostingClassifier:<br>
>HistGradientBoostingClassifier is a machine learning algorithm that uses a gradient boosting technique to fit a history
- KBinsDiscretizer:<br>
>KBinsDiscretizer is a class that is used to discretize continuous features into a fixed number
- make_pipeline:<br>
>It is a function that takes in a list of classifiers and returns a pipeline object. The pipeline object
- SplineTransformer:<br>
>SplineTransformer is a class that transforms the input data into a new representation using splines. It
- train_test_split:<br>
>train_test_split() is a function in scikit-learn that is used to split a dataset
- pd:<br>
>It is a module that provides a high-performance, easy-to-use data structure for tabular data.
- DecisionBoundaryDisplay:<br>
>The DecisionBoundaryDisplay class is a tool for visualizing the decision boundary of a classifier. It is
- cm:<br>
>cm is a scalar map that is used to represent the colorbar. It is used to represent the
- plt:<br>
>plt is a Python library that provides a wide range of plotting functions and tools for creating and customizing
- datasets:<br>
>iris
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
- X:<br>
>X is a matrix of size n x 4, where n is the number of samples. Each
- q:<br>
>It is the number of components in the PLS regression model. The higher the value of q,
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. It is
- print:<br>
>The print function is used to display the output of a Python script to the console. It is a
- n:<br>
>n is the number of samples in the dataset.
- B:<br>
>Variable B is a matrix of size (q, p) where q is the number of components and
- Y:<br>
>Y is a matrix of size n x q where n is the number of samples and q is the
- pls2:<br>
>pls2 is a variable that is used to predict the value of B in the equation y = mx
- X_train_r:<br>
>X_train_r is a matrix of reduced dimensionality that contains the principal components of the training data.
- Y_train_r:<br>
>Y_train_r is a matrix of dimension 1000 x 2 that contains the results of the
- X_test_r:<br>
>X_test_r is a matrix of the test data, which is used to calculate the correlation between the
- Y_test_r:<br>
>Y_test_r is the transformed version of the test data, which is the output of the PLSC
- y:<br>
>The variable y is the target variable in the script. It is used to predict the value of the
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
- y_unique:<br>
>y_unique is a list of unique values in the y_test variable. It is used to create a
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
CONTEXT:   Plot classification probability  This example illustrates the use of :class:`sklearn.inspection.DecisionBoundaryDisplay` to plot the
predicted class probabilities of various classifiers in a 2D feature space, mostly for didactic purposes.  The first three columns shows the predicted
probability for varying values of the two features. Round markers represent the test data that was predicted to belong to that class.  In the last
column, all three classes are represented on each plot; the class with the highest predicted probability at each point is plotted. The round markers
show the test data and are colored by their true label. Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause   COMMENT:
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
```

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

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with multivariate response, a.k.a. PLS2   COMMENT:
```python
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Scatter plot of scores   COMMENT:
```python
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], name="train", marker="o", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], name="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 1: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
plt.subplot(224)
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], name="train", marker="o", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], name="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 2: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
```

### notebooks/plot_classification_probability.ipynb
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

## Code Concatenation
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
import numpy as np
n = 500
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
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], name="train", marker="o", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], name="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 1: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
plt.subplot(224)
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], name="train", marker="o", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], name="test", marker="o", s=25)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title(
    "Comp. 2: X vs Y (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 1], Y_test_r[:, 1])[0, 1]
)
plt.xticks(())
plt.yticks(())
plt.legend(loc="best")
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
