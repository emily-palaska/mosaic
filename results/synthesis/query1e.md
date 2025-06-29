# Embedding Code Synthesis
Query `Create a regression model.`
## Script Variables
- make_pipeline:<br>
>The variable make_pipeline is a function that takes a list of classifiers and returns a pipeline object that can
- classifiers:<br>
>The variable classifiers are used to classify the input data into different classes. They are used to identify the
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
- train_test_split:<br>
>train_test_split is a function that splits the dataset into two parts, one part is used for
- pd:<br>
>pd is a python package that provides data structures and operations for manipulating numerical data and time series data
- mpl:<br>
>mpl is a python package that provides a comprehensive set of tools for creating and manipulating plots in a variety
- plt:<br>
>plt is a Python library that is used to create plots. It is a part of the Matplotlib
- accuracy_score:<br>
>Accuracy score is a measure of the quality of a binary classification model. It is calculated as the ratio
- datasets:<br>
>The variable datasets are the datasets used to train the machine learning models. They are used to predict the
- cm:<br>
>cm is a colormap object which is used to color the data points in the scatter plot.
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- DecisionBoundaryDisplay:<br>
>It is a class that is used to display the decision boundary of a classifier. It takes in a
- log_loss:<br>
>The log_loss function is used to calculate the loss function for a binary classification problem. It is a
- roc_auc_score:<br>
>The roc_auc_score is a function that calculates the area under the receiver operating characteristic curve (ROC)
- print:<br>
>The variable print is used to print the correlation matrix of the input data. It is used to check
- q:<br>
>The variable q is the number of components used in the PLS regression model. It is used to
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- Y:<br>
>Y is a matrix of size (n, 4) where n is the number of samples.
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. PLS
- X:<br>
>X is a numpy array containing the training data for the dataset. It is a 2D array
- pls2:<br>
>pls2 is a PLSRegression object that is used to fit the data and predict the output.
- B:<br>
>B is a matrix of size (q, p) where q is the number of components and p
- Y_test_r:<br>
>Y_test_r is a matrix of the same size as X_test, where each row is the result
- y:<br>
>The variable y is a 1000x1 matrix containing the actual values of the dependent variable.
- X_test_r:<br>
>X_test_r is the transformed version of the test dataset using the PLSCanonical model. It
- X_train_r:<br>
>X_train_r is a matrix of 1000 rows and 2 columns. Each row represents a
- Y_train_r:<br>
>Y_train_r is a numpy array of shape (n_samples, 1) which contains the y
- ax:<br>
>ax is a scatter plot object that is used to plot the training data points on the scatter plot.
- StandardScaler:<br>
>StandardScaler is a class that is used to scale the data to a standard normal distribution. It is
- names:<br>
>X
- X_train:<br>
>X_train is a matrix of size 500x2 containing the features of the training dataset.
- X_test:<br>
>X_test is a 2D array containing the test data. It is used to plot the scatter
- len:<br>
>len is a function that returns the length of an object. In this case, it is used to
- clf:<br>
>clf is a classifier object that is used to predict the class of a given data point. It is
- y_train:<br>
>It is a vector of size 400 which contains the labels of the training data. It is used
- i:<br>
>The variable i is a counter that is used to keep track of the number of plots that have been
- y_test:<br>
>It is a test dataset used to evaluate the performance of the classifier. It is a binary classification problem
- zip:<br>
>The zip() function in Python is used to create an iterator that aggregates elements from two or more iter
- score:<br>
>It is a variable that represents the accuracy of the model on the test data. It is used to
- name:<br>
>The variable name is 'clf' which is an abbreviation for classifier. It is a classifier that is
## Synthesis Blocks
### notebooks/dataset2/classification/plot_classification_probability.ipynb
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

### notebooks/dataset2/classification/plot_lda_qda.ipynb
CONTEXT: We generate three datasets. In the first dataset, the two classes share the same covariance matrix, and this covariance matrix has the
specificity of being spherical (isotropic). The second dataset is similar to the first one but does not enforce the covariance to be spherical.
Finally, the third dataset has a non-spherical covariance matrix for each class.   COMMENT:
```python
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Canonical (symmetric) PLS   Transform data   COMMENT:
```python
import matplotlib.pyplot as plt
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
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test", marker="o", s=25)
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
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test", marker="o", s=25)
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

### notebooks/dataset2/classification/plot_classifier_comparison.ipynb
CONTEXT:   Classifier comparison  A comparison of several classifiers in scikit-learn on synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these
examples does not necessarily carry over to real datasets.  Particularly in high-dimensional spaces, data can more easily be separated linearly and
the simplicity of classifiers such as naive Bayes and linear SVMs might lead to better generalization than is achieved by other classifiers.  The
plots show training points in solid colors and testing points semi-transparent. The lower right shows the classification accuracy on the test set.
COMMENT: iterate over classifiers
```python
for name, clf in zip(names, classifiers):
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    )
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
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test", marker="o", s=25)
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
plt.scatter(X_train_r[:, 1], Y_train_r[:, 1], label="train", marker="o", s=25)
plt.scatter(X_test_r[:, 1], Y_test_r[:, 1], label="test", marker="o", s=25)
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
for name, clf in zip(names, classifiers):
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
    )
```
