# Reverse Embedding Code Synthesis
Query `Create Regression, SVM, Tree, AdaBoost and Bayes classifiers. Compare the classifiers and plot them.`
## Script Variables
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
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
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
- Y:<br>
>Y is a matrix of size (n, 4) where n is the number of samples.
- X:<br>
>X is a matrix of size n x q where n is the number of samples and q is the
- plt:<br>
>plt is a Python library that is used to create plots. It is a part of the Matplotlib
## Synthesis Blocks
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

## Code Concatenation
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
