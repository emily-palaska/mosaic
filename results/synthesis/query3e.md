# Embedding Code Synthesis
Query `Graph operations`
## Script Variables
- plt:<br>
>plt is a module that provides a number of command-line interfaces for plotting in Python. It is a
- print:<br>
>The print function is used to display the output of a Python expression on the screen. It is a
- y_test:<br>
>The variable y_test is a numpy array containing the true labels of the test data. It is used
- predicted:<br>
>The variable predicted is the predicted value of the image. It is used to determine the classification of the
- disp:<br>
>disp is a confusion matrix which is used to compare the predicted values with the actual values. It is
- metrics:<br>
>Confusion matrix
- clf:<br>
>It is a classifier that is used to predict the class of a given data point.
- X_test:<br>
>X_test is a matrix of size (n_samples, n_features) where n_samples is the number
- Y_test:<br>
>Y_test is a variable that is used to test the performance of the PLSCanonical algorithm on
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- X_train:<br>
>X_train is a dataset of 1000 observations and 20 variables. It is used to train
- l1:<br>
>l1 is a numpy array of size n which is used to generate the latent variables l1 and
- n:<br>
>The value of n is 1000 which is the number of samples in the dataset.
- Y_train:<br>
>Y_train is a matrix of size 1000x1, which contains the target values for the
- l2:<br>
>l2 is a random variable that is generated using the normal distribution with a size of n. It
- Y:<br>
>Y is a matrix of size (n, 4) where n is the number of samples.
- X:<br>
>X is a matrix of size n x q where n is the number of samples and q is the
- latents:<br>
>latents is a matrix of 4 rows and n columns. Each row represents a latent variable.
- y:<br>
>The variable y is a 1000x1 matrix containing the actual values of the dependent variable.
- p:<br>
>p is the number of components in the PLS model. In this case, it is 3
- pls1:<br>
>pls1 is a PLSRegression object which is used to perform PLS regression. PLS regression
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. PLS
- make_pipeline:<br>
>make_pipeline() is a function in scikit-learn that allows us to create a pipeline of machine
- classifiers:<br>
>The variable classifiers are used to determine the number of classifiers used in the model. This is done by
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
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Canonical (symmetric) PLS   Transform data   COMMENT:
```python
import matplotlib.pyplot as plt
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
subset.   COMMENT: Predict the value of the digit on the test subset
```python
predicted = clf.predict(X_test)
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with multivariate response, a.k.a. PLS2   COMMENT: compare pls2.coef_ with B
```python
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(X, y)
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Dataset based latent variables model   COMMENT:
```python
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)
latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))
X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT: We can also plot a `confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.   COMMENT:
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT: We can also plot a `confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.   COMMENT:
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
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

## Code Concatenation
```python
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
predicted = clf.predict(X_test)
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
pls1 = PLSRegression(n_components=3)
pls1.fit(X, y)
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)
latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))
X_train = X[: n // 2]
Y_train = Y[: n // 2]
X_test = X[n // 2 :]
Y_test = Y[n // 2 :]
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
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
