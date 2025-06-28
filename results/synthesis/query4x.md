# Exhaustive Code Synthesis
Query `How to perform cross_decomposition`
## Script Variables
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
- n:<br>
>n is the number of samples in the dataset.
- pls1:<br>
>pls1 is an instance of the PLSRegression class. PLSRegression is a regression model that
- print:<br>
>The print function is used to display the output of a Python script to the console. It is a
- y:<br>
>y is a dependent variable that represents the target output of the model. It is used to evaluate the
- X_train:<br>
>X_train is a dataset of 10,000 rows and 3 columns. The first column represents
- pca:<br>
>pca is a PCA object which is used to perform Principal Component Analysis. PCA is a dimensionality
- PCA:<br>
>PCA is a dimensionality reduction technique that projects the data onto a lower-dimensional space while preserving the maximum
- fig:<br>
>fig is a variable that is used to plot the data points in the scatter plot. It is used
- y_test:<br>
>y_test is a numpy array of size (1000,) which is the test data set for the
- rng:<br>
>The variable rng is a random number generator used to generate random numbers for the train-test split and the
- plt:<br>
>plt is a python library that is used to create plots in python. It is a part of the
- y_train:<br>
>The variable y_train is a vector of the dependent variable values for the training data. It is used
- LinearRegression:<br>
>It is a machine learning algorithm that is used to fit a linear model with a least squares approach.
- make_pipeline:<br>
>The make_pipeline() function is used to create a pipeline of several steps. The pipeline is a sequence
- X:<br>
>X is a matrix of size (n_samples, n_features) where n_samples is the number of
- pls:<br>
>It is a variable that is used to store the result of the PLS regression model. The P
- X_test:<br>
>X_test is a numpy array containing the test data. It is used to predict the values of y
- axes:<br>
>The variable axes is a two-dimensional array that represents the projection of the data onto the first and second
- train_test_split:<br>
>The train_test_split function is used to split the dataset into training and testing sets. It takes in
- pcr:<br>
>The variable pcr is a function that calculates the r-squared value of a given dataset. It
- PLSRegression:<br>
>PLSRegression is a Python library that implements Partial Least Squares Regression (PLSR). It is
- StandardScaler:<br>
>The StandardScaler is a class that is used to scale the data to a standard normal distribution. It
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

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with univariate response, a.k.a. PLS1   COMMENT:
```python
print("Estimated betas")
print(np.round(pls1.coef_, 1))
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  Projection on one component and predictive power  We now create two regressors: PCR and PLS, and for our illustration purposes we set the
number of components to 1. Before feeding the data to the PCA step of PCR, we first standardize it, as recommended by good practice. The PLS estimator
has built-in scaling capabilities.  For both models, we plot the projected data onto the first component against the target. In both cases, this
projected data is what the regressors will use as training data.   COMMENT: retrieve the PCA step of the pipeline
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)
pca = pcr.named_steps["pca"]

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(pca.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[0].scatter(
    pca.transform(X_test), pcr.predict(X_test), alpha=0.3, label="predictions"
)
axes[0].set(
    xlabel="Projected data onto first PCA component", ylabel="y", title="PCR / PCA"
)
axes[0].legend()
axes[1].scatter(pls.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[1].scatter(
    pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="predictions"
)
axes[1].set(xlabel="Projected data onto first PLS component", ylabel="y", title="PLS")
axes[1].legend()
plt.tight_layout()
plt.show()
```

## Code Concatenation
```python
import numpy as np
n = 500
print("Estimated betas")
print(np.round(pls1.coef_, 1))
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)
pca = pcr.named_steps["pca"]

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].scatter(pca.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[0].scatter(
    pca.transform(X_test), pcr.predict(X_test), alpha=0.3, label="predictions"
)
axes[0].set(
    xlabel="Projected data onto first PCA component", ylabel="y", title="PCR / PCA"
)
axes[0].legend()
axes[1].scatter(pls.transform(X_test), y_test, alpha=0.3, label="ground truth")
axes[1].scatter(
    pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="predictions"
)
axes[1].set(xlabel="Projected data onto first PLS component", ylabel="y", title="PLS")
axes[1].legend()
plt.tight_layout()
plt.show()
```
