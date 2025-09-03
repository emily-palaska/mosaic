# Random Code Synthesis
Query `Compute Wikipedia principal eigenvector for PageRank.`
## Script Variables
- X:<br>
>X is a list of tuples containing the dataset and the algorithm parameters.
- ica:<br>
>ica is a FastICA object which is used to perform Independent Component Analysis (ICA) on the
- S_ica_:<br>
>The variable S_ica_ is a matrix that contains the components of the Independent Component Analysis (ICA
- time:<br>
>The variable time is a numpy array that contains 2000 values ranging from 0 to 8
- np:<br>
>Numpy is a Python library that provides a multidimensional array object, which can hold data of different
- s1:<br>
>The variable s1 is a sine function of time. It is used to generate a sine wave that
- label:<br>
>The variable label is a unique identifier for each observation in the dataset. It is used to identify the
- setting:<br>
>The variable setting is used to specify the parameters of the gradient boosting classifier. It includes the number of
- labels:<br>
>labels
- train_test_split:<br>
>The train_test_split function is used to split the data into training and testing sets. It takes in
- y_train:<br>
>y_train is the dependent variable in the dataset. It is the target variable that we want to predict
- ensemble:<br>
>The variable ensemble is a set of variables that are used to predict the output of a machine learning algorithm
- X_train:<br>
>X_train is a numpy array of shape (n_samples, n_features) containing the training data.
- plt:<br>
>plt is a module that is used to create plots in Python. It is a part of the Mat
- clf:<br>
>clf is a classifier that is used to predict the probability of a positive class. It is a calibrated
- dict:<br>
>dict is a python dictionary which is a collection of key-value pairs.
- original_params:<br>
>It is a dictionary containing the parameters for the gradient boosting classifier. The parameters are
- y_test:<br>
>It is the test set of the data that is used to evaluate the performance of the model. It
- y:<br>
>y is a target variable which is used to identify the type of iris flower. It is a categorical
- color:<br>
>The variable color is a string that represents the color of the points in the scatter plot. It is
- X_test:<br>
>X_test is a numpy array containing the test data. It is used to predict the values of y
- sample_weight:<br>
>Sample weight is a variable that is used to indicate the importance of each sample in the dataset. It
- sw_train:<br>
>sw_train is a variable that is used to calculate the weights of the training data. It is used
- prob_pos_isotonic:<br>
>Prob_pos_isotonic is a variable that contains the probability of a positive class for each sample
- CalibratedClassifierCV:<br>
>CalibratedClassifierCV is a class that calibrates a classifier using isotonic regression. It is
- clf_isotonic:<br>
>It is a classifier that uses the isotonic method to calibrate the predictions of the classifier clf.
- pca:<br>
>PCA is a dimensionality reduction technique that projects the data onto a lower dimensional space while retaining as much
- var:<br>
>var is a variable that is used to multiply the value of comp by the value of var.
- comp:<br>
>The variable comp is a list of tuples, where each tuple represents a component of the principal components of
- enumerate:<br>
>The enumerate() function returns a list of tuples where the first element of each tuple is the index of
- cov:<br>
>The variable cov is a 2x2 matrix that represents the covariance between the two features in the
- rng:<br>
>The variable rng is a random number generator that is used to generate random numbers for the train-test split
- i:<br>
>The variable i is a placeholder for the number of components to be used in the PCA or Incremental
- n_samples:<br>
>n_samples is the number of samples in the dataset. It is used to generate random noise in the
- zip:<br>
>The variable zip is a function that takes two iterables and returns a list of tuples, where the
- mse:<br>
>mse is the mean squared error of the predictions made by the model on the test set. It is
- mean_squared_error:<br>
>Mean squared error is a measure of the average of the squares of the differences between the values predicted by
- reg:<br>
>reg is a regression object that is used to fit a model to the training data. It is used
- print:<br>
>The print function is used to display the output of the script. It is a built-in function in
- IncrementalPCA:<br>
>The IncrementalPCA class is a subclass of the PCA class that can be used to perform dimensionality
- X_transformed:<br>
>X_transformed is a numpy array containing the transformed data from the original dataset. It is used to
- iris:<br>
>The iris dataset is a set of 150 observations of 4 variables, each of which is a
- target_name:<br>
>The target_name variable is a list of strings that represent the names of the three different species of Iris
- X_ipca:<br>
>X_ipca is a numpy array that contains the transformed data from the IncrementalPCA class. It
- title:<br>
>The variable title is a string that describes the role and significance of the variable within the script. It
- ipca:<br>
>ipca is an instance of IncrementalPCA class. It is used to perform dimensionality reduction on
- X_pca:<br>
>X_pca is a matrix of dimension (n_samples, n_components) where n_samples is the
- load_iris:<br>
>load_iris() is a function in the sklearn.datasets module that loads the iris dataset into memory.
- n_components:<br>
>n_components is the number of components to keep in the PCA or Incremental PCA transformation.
- err:<br>
>The variable err is used to calculate the mean absolute unsigned error between the two transformed datasets. It is
- colors:<br>
>The variable colors is used to represent the different classes of the iris dataset. It is a list of
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  The data  We start by creating a simple dataset with two features. Before we even dive into PCR and PLS, we fit a PCA estimator to display
the two principal components of this dataset, i.e. the two directions that explain the most variance in the data.   COMMENT: scale component by its
variance explanation power
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import pca
rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = pca(n_components=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var

    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)
plt.legend()
plt.show()
```

### notebooks/dataset2/decomposition/plot_incremental_pca.ipynb
CONTEXT:   Incremental PCA  Incremental principal component analysis (IPCA) is typically used as a replacement for principal component analysis (PCA)
when the dataset to be decomposed is too large to fit in memory. IPCA builds a low-rank approximation for the input data using an amount of memory
which is independent of the number of input data samples. It is still dependent on the input data features, but changing the batch size allows for
control of memory usage.  This example serves as a visual check that IPCA is able to find a similar projection of the data to PCA (to a sign flip),
while only processing a few samples at a time. This can be considered a "toy example", as IPCA is intended for large datasets which do not fit in main
memory, requiring incremental approaches.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import pca, IncrementalPCA
iris = load_iris()
X = iris.data
y = iris.target
n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)
pca = pca(n_components=n_components)
X_pca = pca.fit_transform(X)
colors = ["navy", "turquoise", "darkorange"]
for X_transformed, title in [(X_ipca, "Incremental pca"), (X_pca, "pca")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[y == i, 0],
            X_transformed[y == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )
    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])
plt.show()
```

### notebooks/dataset2/clustering/plot_kmeans_assumptions.ipynb
CONTEXT:  Data generation  The function :func:`~sklearn.datasets.make_blobs` generates isotropic (spherical) gaussian blobs. To obtain anisotropic
(elliptical) gaussian blobs one has to define a linear `transformation`.   COMMENT: Unevenly sized blobs
```python
)
```

### notebooks/dataset2/clustering/plot_linkage_comparison.ipynb
CONTEXT: Run the clustering and plot   COMMENT: normalize dataset for easier parameter selection
```python
    X = StandardScaler().fit_transform(X)
```

### notebooks/dataset2/decomposition/plot_ica_vs_pca.ipynb
CONTEXT:  Generate sample data   COMMENT: Estimate the sources
```python
S_ica_ = ica.fit(X).transform(X)
```

### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Signal 1 : sinusoidal signal
```python
s1 = np.sin(2 * time)
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_regularization.ipynb
CONTEXT:   Gradient Boosting regularization  Illustration of the effect of different regularization strategies for Gradient Boosting. The example is
taken from Hastie et al 2009 [1]_.  The loss function used is binomial deviance. Regularization via shrinkage (``learning_rate < 1.0``) improves
performance considerably. In combination with shrinkage, stochastic gradient boosting (``subsample < 1.0``) can produce more accurate models by
reducing the variance via bagging. Subsampling without shrinkage usually does poorly. Another strategy to reduce the variance is by subsampling the
features analogous to the random splits in Random Forests (via the ``max_features`` parameter).  .. [1] T. Hastie, R. Tibshirani and J. Friedman,
"Elements of Statistical     Learning Ed. 2", Springer, 2009.  COMMENT: map labels from {-1, 1} to {0, 1}
```python
labels, y = np.unique(y, return_inverse=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
original_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
}
plt.figure()
for label, color, setting in [
    ("No shrinkage", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
    ("learning_rate=0.2", "turquoise", {"learning_rate": 0.2, "subsample": 1.0}),
    ("subsample=0.5", "blue", {"learning_rate": 1.0, "subsample": 0.5}),
    (
        "learning_rate=0.2, subsample=0.5",
        "gray",
        {"learning_rate": 0.2, "subsample": 0.5},
    ),
    (
        "learning_rate=0.2, max_features=2",
        "magenta",
        {"learning_rate": 0.2, "max_features": 2},
    ),
]:
    original_params = dict(original_params)
    original_params.update(setting)
    clf = ensemble.GradientBoostingClassifier(**original_params)
    clf.fit(X_train, y_train)
```

### notebooks/dataset2/calibration/plot_calibration.ipynb
CONTEXT:  Gaussian Naive-Bayes   COMMENT: With no calibration
```python
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_regression.ipynb
CONTEXT:  Fit regression model  Now we will initiate the gradient boosting regressors and fit it with our training data. Let's also look and the mean
squared error on the test data.   COMMENT:
```python
reg = ensemble.GradientBoostingRegressor(**original_params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import pca
rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = pca(n_components=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var

    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import pca, IncrementalPCA
iris = load_iris()
X = iris.data
y = iris.target
n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)
pca = pca(n_components=n_components)
X_pca = pca.fit_transform(X)
colors = ["navy", "turquoise", "darkorange"]
for X_transformed, title in [(X_ipca, "Incremental pca"), (X_pca, "pca")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(
            X_transformed[y == i, 0],
            X_transformed[y == i, 1],
            color=color,
            lw=2,
            label=target_name,
        )
    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
    else:
        plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])
plt.show()
)
    X = StandardScaler().fit_transform(X)
S_ica_ = ica.fit(X).transform(X)
s1 = np.sin(2 * time)
labels, y = np.unique(y, return_inverse=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
original_params = {
    "n_estimators": 400,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": 2,
    "min_samples_split": 5,
}
plt.figure()
for label, color, setting in [
    ("No shrinkage", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
    ("learning_rate=0.2", "turquoise", {"learning_rate": 0.2, "subsample": 1.0}),
    ("subsample=0.5", "blue", {"learning_rate": 1.0, "subsample": 0.5}),
    (
        "learning_rate=0.2, subsample=0.5",
        "gray",
        {"learning_rate": 0.2, "subsample": 0.5},
    ),
    (
        "learning_rate=0.2, max_features=2",
        "magenta",
        {"learning_rate": 0.2, "max_features": 2},
    ),
]:
    original_params = dict(original_params)
    original_params.update(setting)
    clf = ensemble.GradientBoostingClassifier(**original_params)
    clf.fit(X_train, y_train)
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
reg = ensemble.GradientBoostingRegressor(**original_params)
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
```
