# Embedding Code Synthesis
Query `Graph operations`
## Script Variables
- metrics:<br>
>The variable metrics is a function that calculates the classification report for a given classifier. It takes two arguments
- predicted:<br>
>The variable predicted is a variable that is used to predict the output of the model. It is a
- y_test:<br>
>y_test is a numpy array that contains the actual values of the test data. It is used to
- disp:<br>
>disp is a confusion matrix that is used to compare the predicted values with the actual values.
- plt:<br>
>plt is a module in python that is used to create plots. It is used in this script to
- print:<br>
>print() is a function that prints a string to the console. In this case, it is used
- X_test:<br>
>X_test is a 2D numpy array containing the test data. It is a matrix of shape
- clf:<br>
>clf is a classifier object which is used to predict the class of the test data.
- X:<br>
>X is a matrix of size n x 4, where n is the number of samples. Each
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
- Y_train:<br>
>Y_train is a variable that contains the target values for the training data set. It is used to
- n:<br>
>n is the number of samples in the dataset.
- Y_test:<br>
>Y_test is a variable that is used to test the correlation between the two latent variables. It is
- Y:<br>
>Y is a matrix of size n x q where n is the number of samples and q is the
- X_train:<br>
>X_train is a matrix of size (n_samples, n_features) where n_samples is the number
- l2:<br>
>l2 is a random normal variable with mean 0 and standard deviation 1. It is used
- latents:<br>
>latents is a 2D array of size (n, 4) where each row represents
- l1:<br>
>l1 is a numpy array of size n, which is a random normal distribution with mean 0
- PLSRegression:<br>
>PLSRegression is a class that implements Partial Least Squares (PLS) regression. It is
- p:<br>
>p is a constant value that is used to determine the number of columns in the dataset.
- y:<br>
>The variable y is the target variable in the script. It is used to predict the value of the
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
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Canonical (symmetric) PLS   Transform data   COMMENT:
```python
import matplotlib.pyplot as plt
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

### notebooks/plot_digits_classification.ipynb
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
PLSRegression = PLSRegression(n_components=3)
PLSRegression.fit(X, y)
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

### notebooks/plot_digits_classification.ipynb
CONTEXT: We can also plot a `confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.   COMMENT:
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
```

### notebooks/plot_digits_classification.ipynb
CONTEXT: We can also plot a `confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.   COMMENT:
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
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
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.inspection import DecisionBoundaryDisplay
predicted = clf.predict(X_test)
n = 1000
p = 10
X = np.random.normal(size=n * p).reshape((n, p))
y = X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n * 1) + 5
PLSRegression = PLSRegression(n_components=3)
PLSRegression.fit(X, y)
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
