# Embedding Code Synthesis
Query `Plot predicted probabilities vs true outcomes.`
## Script Variables
- X_test:<br>
>X_test is a test dataset that is used to evaluate the performance of the model. It is created
- X_train:<br>
>X_train is a matrix of size (n_samples, n_features) where n_samples is the number
- y_train:<br>
>It is a vector of labels that indicate the class of each sample in the training set.
- X:<br>
>X is a 2D array of size (n_samples, n_features) where n_samples is
- y_test:<br>
>The variable y_test is a list of labels that represent the actual class of the test samples. It
- train_test_split:<br>
>The train_test_split function is used to split the dataset into training and testing sets. The training set
- y:<br>
>The variable y is a numpy array of shape (n_samples,) containing the labels of the samples.
- make_classification:<br>
>The make_classification function is used to generate a classification dataset. It takes in a number of parameters such
- linalg:<br>
>The variable linalg is used to perform linear algebra operations such as matrix multiplication, matrix inversion, and
- Pipeline:<br>
>Pipeline is a class that allows us to create a sequence of steps in a pipeline. Each step is
- GridSearchCV:<br>
>GridSearchCV is a class that is used to perform a grid search over a parameter space. It
- feature_selection:<br>
>Feature selection is a process of selecting a subset of features from a dataset that are most relevant to the
- tempfile:<br>
>Temporarily stores data for later use. In this case, the data is stored in a temporary directory
- np:<br>
>The variable np is a Python package that provides a large collection of mathematical functions and data structures for scientific
- FeatureAgglomeration:<br>
>FeatureAgglomeration is a Python module that implements a hierarchical clustering algorithm called Ward's method.
- KFold:<br>
>Kfold is a class that splits the data into k folds. It is used to split the data
- grid_to_graph:<br>
>Grid_to_graph is a function that creates a connectivity matrix for a given grid. The connectivity matrix is
- plt:<br>
>plt is a variable that is used to create a plot. It is a module that is used to
- ndimage:<br>
>ndimage is a module in Python that is used to perform image processing tasks. It is part of
- shutil:<br>
>shutil is a module in Python that provides a number of functions for working with files and directories.
- Memory:<br>
>Memory is a class that is used to cache the results of the BayesianRidge model. It is
- BayesianRidge:<br>
>BayesianRidge is a regression model that uses a Bayesian approach to estimate the coefficients of a linear
- shrinkages:<br>
>Shrinkages is a list of values that represent the shrinkage parameter for each model. The shrink
- ShrunkCovariance:<br>
>It is a class that implements the Shrinkage Covariance estimator. It takes a shrinkage parameter
- tuned_parameters:<br>
>The variable tuned_parameters is a dictionary that contains the parameters that will be optimized during the grid search.
- cv:<br>
>cv is a cross-validation object that contains the best estimator and the best score for the given data.
- gs:<br>
>gs is a 2D numpy array that is used to create a grid of subplots in a
- i:<br>
>The variable i is used to iterate through the list of classifiers. It is used to access the corresponding
- calibration_displays:<br>
>Calibration displays are used to evaluate the performance of a model in predicting the probability of a given outcome
- col:<br>
>col is the column of the grid that the subplot is placed in. It is used to create the
- fig:<br>
>fig is a variable that is used to store the figure object. It is used to create a plot
- name:<br>
>The variable name is 'clf' which is a classifier. It is used to fit the training data
- clf_list:<br>
>It is a list of tuples, where each tuple contains a classifier and its name. The list is
- enumerate:<br>
>The enumerate function is used to return the index of an iterable along with the value of the element.
- _:<br>
>It is a tuple that contains the row and column coordinates of the subplot where the histogram will be displayed
- ax:<br>
>The variable ax is a subplot object that is used to create a histogram of the y_prob values for
- colors:<br>
>Colors are used to represent different classes in the histogram. The colors are chosen randomly from a list of
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_calibration_curve.ipynb
CONTEXT:   Probability Calibration curves  When performing classification one often wants to predict not only the class label, but also the associated
probability. This probability gives some kind of confidence on the prediction. This example demonstrates how to visualize how well calibrated the
predicted probabilities are using calibration curves, also known as reliability diagrams. Calibration of an uncalibrated classifier will also be
demonstrated.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)
```

### notebooks/dataset2/feature_selection/plot_feature_selection_pipeline.ipynb
CONTEXT:   Pipeline ANOVA SVM  This example shows how a feature selection can be easily integrated within a machine learning pipeline.  We also show
that you can easily inspect part of the pipeline.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

### notebooks/dataset2/clustering/plot_feature_agglomeration_vs_univariate_selection.ipynb
CONTEXT:   Feature agglomeration vs. univariate selection  This example compares 2 dimensionality reduction strategies:  - univariate feature
selection with Anova  - feature agglomeration with Ward hierarchical clustering  Both methods are compared in a regression problem using a
BayesianRidge as supervised estimator.  COMMENT:
```python
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy import linalg, ndimage
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
```

### notebooks/dataset2/covariance_estimation/plot_covariance_estimation.ipynb
CONTEXT:  Compare different approaches to setting the regularization parameter  Here we compare 3 approaches:  * Setting the parameter by cross-
validating the likelihood on three folds   according to a grid of potential shrinkage parameters.  * A close formula proposed by Ledoit and Wolf to
compute   the asymptotically optimal regularization parameter (minimizing a MSE   criterion), yielding the :class:`~sklearn.covariance.LedoitWolf`
covariance estimate.  * An improvement of the Ledoit-Wolf shrinkage, the   :class:`~sklearn.covariance.OAS`, proposed by Chen et al. Its   convergence
is significantly better under the assumption that the data   are Gaussian, in particular for small samples.   COMMENT:
```python
tuned_parameters = [{"shrinkage": shrinkages}]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
cv.fit(X_train)
```

### notebooks/dataset2/calibration/plot_compare_calibration.ipynb
CONTEXT:  Calibration curves  Below, we train each of the four models with the small training dataset, then plot calibration curves (also known as
reliability diagrams) using predicted probabilities of the test dataset. Calibration curves are created by binning predicted probabilities, then
plotting the mean predicted probability in each bin against the observed frequency ('fraction of positives'). Below the calibration curve, we plot a
histogram showing the distribution of the predicted probabilities or more specifically, the number of samples in each predicted probability bin.
COMMENT: Add histogram
```python
_ = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    _, col = _[i]
    ax = fig.add_subplot(gs[_, col])
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
```

## Code Concatenation
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy import linalg, ndimage
from sklearn import feature_selection
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
tuned_parameters = [{"shrinkage": shrinkages}]
cv = GridSearchCV(ShrunkCovariance(), tuned_parameters)
cv.fit(X_train)
_ = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    _, col = _[i]
    ax = fig.add_subplot(gs[_, col])
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
```
