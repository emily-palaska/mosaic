# Reverse Embedding Code Synthesis
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
- X:<br>
>X is a matrix of size n x 4, where n is the number of samples. Each
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
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
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  PLS regression, with multivariate response, a.k.a. PLS2   COMMENT:
```python
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
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
Y = np.dot(X, B) + np.random.normal(size=n * q).reshape((n, q)) + 5
pls2 = PLSRegression(n_components=3)
pls2.fit(X, Y)
print("True B (such that: Y = XB + Err)")
print(B)
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
