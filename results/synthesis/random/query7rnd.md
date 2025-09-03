# Random Code Synthesis
Query `Analyze forest embedding of iris dataset.`
## Script Variables
- fig:<br>
>The fig variable is a matplotlib figure object that is used to create a 3D plot of the
- ax:<br>
>ax is a matplotlib axis object. It is used to plot the graph. It is an instance of
- y:<br>
>The variable y is the target variable of the dataset. It is the variable that we want to predict
- _:<br>
>The variable _ is a placeholder for the number of iterations of the gradient boosting algorithm. It is used
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses historical data to predict future values. It
- plt:<br>
>plt is a python library that is used to create plots in python. It is a part of the
- PartialDependenceDisplay:<br>
>PartialDependenceDisplay is a class used to display the partial dependence of a model on a given
- hgbt_cst:<br>
>It is a variable that is used to describe the role and significance of the HistGradientBoostingRegressor
- X:<br>
>X is a 2D array containing the data points of the dataset. It is a random sample
- monotonic_cst:<br>
>It is a dictionary that contains the index of the features that are monotonic with respect to the target
- disp:<br>
>disp is a variable that is used to display the partial dependence of the model on the given features.
- cat_selector:<br>
>It is a function that takes in a list of column names and returns a column transformer that can be
- make_column_transformer:<br>
>The make_column_transformer function is used to create a column transformer object that can be used to transform
- cat_tree_processor:<br>
>It is a pipeline that is used to preprocess the categorical data. The pipeline consists of two steps
- make_pipeline:<br>
>The make_pipeline function is used to create a pipeline of estimators. The pipeline is a sequence of
- num_selector:<br>
>num_selector is a function that takes a pandas dataframe as input and returns a column selector object. The
- num_tree_processor:<br>
>The num_tree_processor is a SimpleImputer object that is used to impute missing values in numerical
- SimpleImputer:<br>
>SimpleImputer is a class in sklearn.preprocessing that is used to impute missing values in numerical data
- tree_preprocessor:<br>
>Tree Preprocessor is a pipeline that takes a tree as input and returns a new tree with the same
- OrdinalEncoder:<br>
>OrdinalEncoder is a class used for encoding categorical data into numerical values. It is used for converting categorical
- X_test:<br>
>X_test is a matrix of 2 columns and 100 rows. It contains the values of the
- clf:<br>
>clf is a Random Forest Classifier. It is a supervised machine learning algorithm that uses a decision tree to
- colors:<br>
>The variable colors is a list of colors that are used to represent the different classes in the prediction.
- clf_probs:<br>
>The variable clf_probs is a numpy array that contains the probabilities of the uncalibrated classifier. It
- cal_clf_probs:<br>
>It is a variable that stores the probabilities of the classes predicted by the classifier. The variable is used
- cal_clf:<br>
>CalibratedClassifierCV is a class that calibrates a classifier by fitting a separate calibration model to
- GridSearchCV:<br>
>GridSearchCV is a class that is used to perform a grid search over a parameter space. It
- OAS:<br>
>OAS is a class that is used to perform a logistic regression on the given data. It is
- LedoitWolf:<br>
>LedoitWolf is a class that implements the Ledoit-Wolf shrinkage estimator for covariance matrices
- sample_weight:<br>
>Sample weight is a variable that is used to indicate the importance of each sample in the dataset. It
- y_train:<br>
>It is a variable that contains the target variable of the dataset. It is a vector of integers that
- sw_train:<br>
>sw_train is a variable that is used to calculate the weights of the training data. It is used
- X_train:<br>
>X_train is a numpy array containing the training data. It is used to train the model and make
- train_test_split:<br>
>It is a function that is used to split the dataset into training and testing sets. It takes in
- sw_test:<br>
>sw_test is a variable that is used to split the dataset into training and testing sets. It is
- y_test:<br>
>The variable y_test is a vector of test data that is used to evaluate the performance of the model
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- print:<br>
>The print function is used to display the output of the script to the console. It takes a single
- pls:<br>
>pls is a variable that is used to store the PLS regression model. It is used to predict
- iris:<br>
>iris is a pandas dataframe containing the iris dataset. It has 4 columns
- PCA:<br>
>PCA stands for Principal Component Analysis. It is a dimensionality reduction technique that is used to reduce the
- scatter:<br>
>The scatter variable is a scatter plot that shows the relationship between the features of the iris dataset. The
- X_reduced:<br>
>X_reduced is a 3D array of shape (150, 3) containing the first
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_hgbt_regression.ipynb
CONTEXT: We observe a tendence to over-estimate the energy transfer. This could be be quantitatively confirmed by computing empirical coverage numbers
as done in the `calibration of confidence intervals section <calibration-section>`. Keep in mind that those predicted percentiles are just estimations
from a model. One can still improve the quality of such estimations by:  - collecting more data-points; - better tuning of the model hyperparameters,
see   `sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py`; - engineering more predictive features from the same data, see
`sphx_glr_auto_examples_applications_plot_cyclical_feature_engineering.py`.   Monotonic constraints  Given specific domain knowledge that requires the
relationship between a feature and the target to be monotonically increasing or decreasing, one can enforce such behaviour in the predictions of a
HGBT model using monotonic constraints. This makes the model more interpretable and can reduce its variance (and potentially mitigate overfitting) at
the risk of increasing bias. Monotonic constraints can also be used to enforce specific regulatory requirements, ensure compliance and align with
ethical considerations.  In the present example, the policy of transferring energy from Victoria to New South Wales is meant to alleviate price
fluctuations, meaning that the model predictions have to enforce such goal, i.e. transfer should increase with price and demand in New South Wales,
but also decrease with price and demand in Victoria, in order to benefit both populations.  If the training data has feature names, it’s possible to
specify the monotonic constraints by passing a dictionary with the convention:  - 1: monotonic increase - 0: no constraint - -1: monotonic decrease
Alternatively, one can pass an array-like object encoding the above convention by position.   COMMENT:
```python
from sklearn.inspection import PartialDependenceDisplay
monotonic_cst = {
    "date": 0,
    "day": 0,
    "period": 0,
    "nswdemand": 1,
    "nswprice": 1,
    "vicdemand": -1,
    "vicprice": -1,
}
hgbt_cst = HistGradientBoostingRegressor(
    categorical_features=None, random_state=42
).fit(X, y)
hgbt_cst = HistGradientBoostingRegressor(
    monotonic_cst=monotonic_cst, categorical_features=None, random_state=42
).fit(X, y)
fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
disp = PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    ax=ax[0],
)
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
disp = PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["vicdemand", "vicprice"],
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    ax=ax[1],
)
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["vicdemand", "vicprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
_ = plt.legend()
```

### notebooks/dataset2/ensemble_methods/plot_stack_predictors.ipynb
CONTEXT: Then, we will need to design preprocessing pipelines which depends on the ending regressor. If the ending regressor is a linear model, one
needs to one-hot encode the categories. If the ending regressor is a tree-based model an ordinal encoder will be sufficient. Besides, numerical values
need to be standardized for a linear model while the raw numerical data can be treated as is by a tree-based model. However, both models need an
imputer to handle missing values.  We will first design the pipeline required for the tree-based models.   COMMENT:
```python
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
cat_tree_processor = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2,
)
num_tree_processor = SimpleImputer(strategy="mean", add_indicator=True)
tree_preprocessor = make_column_transformer(
    (num_tree_processor, num_selector), (cat_tree_processor, cat_selector)
)
tree_preprocessor
```

### notebooks/dataset2/calibration/plot_calibration_multiclass.ipynb
CONTEXT:  Compare probabilities Below we plot a 2-simplex with arrows showing the change in predicted probabilities of the test samples.   COMMENT:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]
clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)
```

### notebooks/dataset2/covariance_estimation/plot_covariance_estimation.ipynb
CONTEXT:  Compute the likelihood on test data   COMMENT: under the ground-truth model, which we would not have access to in real settings
```python
from sklearn.covariance import OAS, LedoitWolf
from sklearn.model_selection import GridSearchCV
```

### notebooks/dataset2/decomposition/plot_pca_iris.ipynb
CONTEXT: Each data point on each scatter plot refers to one of the 150 iris flowers in the dataset, with the color indicating their respective type
(Setosa, Versicolor, and Virginica).  You can already see a pattern regarding the Setosa type, which is easily identifiable based on its short and
wide sepal. Only considering these two dimensions, sepal width and length, there's still overlap between the Versicolor and Virginica types.  The
diagonal of the plot shows the distribution of each feature. We observe that the petal width and the petal length are the most discriminant features
for the three types.   Plot a PCA representation Let's apply a Principal Component Analysis (PCA) to the iris dataset and then plot the irises across
the first three PCA dimensions. This will allow us to better differentiate among the three types!   COMMENT: unused but required import for doing 3d
projections with matplotlib < 3.2
```python
import mpl_toolkits.mplot3d

from sklearn.decomposition import PCA
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)
ax.set(
    title="First three PCA dimensions",
    xlabel="1st Eigenvector",
    ylabel="2nd Eigenvector",
    zlabel="3rd Eigenvector",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
```

### notebooks/dataset2/calibration/plot_calibration.ipynb
CONTEXT:  Generate synthetic dataset   COMMENT: Generate 3 blobs with 2 classes where the second blob contains half positive samples and half negative
samples. Probability in this blob is therefore 0.5.
```python
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.9, random_state=42
)
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  Projection on one component and predictive power  We now create two regressors: PCR and PLS, and for our illustration purposes we set the
number of components to 1. Before feeding the data to the PCA step of PCR, we first standardize it, as recommended by good practice. The PLS estimator
has built-in scaling capabilities.  For both models, we plot the projected data onto the first component against the target. In both cases, this
projected data is what the regressors will use as training data.   COMMENT:
```python
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```

## Code Concatenation
```python
from sklearn.inspection import PartialDependenceDisplay
monotonic_cst = {
    "date": 0,
    "day": 0,
    "period": 0,
    "nswdemand": 1,
    "nswprice": 1,
    "vicdemand": -1,
    "vicprice": -1,
}
hgbt_cst = HistGradientBoostingRegressor(
    categorical_features=None, random_state=42
).fit(X, y)
hgbt_cst = HistGradientBoostingRegressor(
    monotonic_cst=monotonic_cst, categorical_features=None, random_state=42
).fit(X, y)
fig, ax = plt.subplots(nrows=2, figsize=(15, 10))
disp = PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    ax=ax[0],
)
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["nswdemand", "nswprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
disp = PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["vicdemand", "vicprice"],
    line_kw={"linewidth": 2, "label": "unconstrained", "color": "tab:blue"},
    ax=ax[1],
)
PartialDependenceDisplay.from_estimator(
    hgbt_cst,
    X,
    features=["vicdemand", "vicprice"],
    line_kw={"linewidth": 2, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
_ = plt.legend()
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
cat_tree_processor = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2,
)
num_tree_processor = SimpleImputer(strategy="mean", add_indicator=True)
tree_preprocessor = make_column_transformer(
    (num_tree_processor, num_selector), (cat_tree_processor, cat_selector)
)
tree_preprocessor
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
colors = ["r", "g", "b"]
clf_probs = clf.predict_proba(X_test)
cal_clf_probs = cal_clf.predict_proba(X_test)
from sklearn.covariance import OAS, LedoitWolf
from sklearn.model_selection import GridSearchCV
import mpl_toolkits.mplot3d

from sklearn.decomposition import PCA
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)
ax.set(
    title="First three PCA dimensions",
    xlabel="1st Eigenvector",
    ylabel="2nd Eigenvector",
    zlabel="3rd Eigenvector",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.9, random_state=42
)
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```
