# Exhaustive Code Synthesis
Query `How to perform cross_decomposition`
## Script Variables
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- print:<br>
>The variable print is used to print the correlation matrix of the input data. It is used to check
- pls1:<br>
>pls1 is a PLSRegression object which is used to perform PLS regression. PLS regression
- LinearRegression:<br>
>LinearRegression is a linear regression model that fits a linear model with coefficients w to minimize the residual sum
- axes:<br>
>The variable axes in the given Python script are as follows
- plt:<br>
>plt is a module in python that is used for plotting graphs. It is a part of the matplotlib
- pca:<br>
>pca is a PCA object that is used to reduce the dimensionality of the data. It does
- pls:<br>
>It is a variable that is used to store the value of the PLS regression score. This score
- y_train:<br>
>The variable y_train is a numpy array containing the target values for the training data. It is used
- fig:<br>
>fig is a figure object that is used to display the results of the PCA and PLS regression models
- pcr:<br>
>The variable pcr is a pipeline that contains a standard scaler, a PCA component, and a linear
- X_train:<br>
>X_train is a numpy array containing the training data. It is used to train the model and make
- y_test:<br>
>The variable y_test is a test dataset that is used to evaluate the performance of the PCA algorithm.
- PCA:<br>
>PCA is a dimensionality reduction technique that is used to reduce the number of features in a dataset while
- X_test:<br>
>X_test is a dataset of 2 components of the PCA transformation of the original dataset X_train.
- rng:<br>
>The variable rng is used to generate random numbers for the train-test split and the PLSRegression model
- X:<br>
>X is a dataset containing information about the properties of a house, such as its size, location,
- PLSRegression:<br>
>PLSRegression is a regression model that uses the partial least squares (PLS) method to find
- make_pipeline:<br>
>The make_pipeline() function is used to create a pipeline of steps. The steps are executed in the
- y:<br>
>The variable y is the dependent variable in the given Python script. It represents the target variable that we
- StandardScaler:<br>
>StandardScaler is a class that is used to scale the features of a dataset. It is a preprocessing
- train_test_split:<br>
>The train_test_split function is used to split the data into training and testing sets. It takes in
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

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with univariate response, a.k.a. PLS1   COMMENT:
```python
print("Estimated betas")
print(np.round(pls1.coef_, 1))
```

## Code Concatenation
```python
import numpy as np
n = 500
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
print("Estimated betas")
print(np.round(pls1.coef_, 1))
```
