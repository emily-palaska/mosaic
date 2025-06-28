# Exhaustive Code Synthesis
Query `Create classifiers with different types and names. Compare the classifiers and plot them.`
## Script Variables
- disp:<br>
>disp is a scalar map object that is used to create a colorbar for the surface plot. It
- ax_single:<br>
>It is a matplotlib Axes object that is used to create a colorbar for the figure. The color
- fig:<br>
>fig is a variable that is used to store the figure object that is created by the script. It
- cm:<br>
>cm is a scalar map that is used to represent the colorbar. It is used to represent the
- plt:<br>
>plt is a Python library that provides a wide range of plotting functions and tools for creating and customizing
- _:<br>
>The variable _ is used to store the color map for the probability class. It is used to create
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
### notebooks/plot_classification_probability.ipynb
CONTEXT:  Plotting the decision boundaries  For each classifier, we plot the per-class probabilities on the first three columns and the probabilities
of the most likely class on the last column.   COMMENT: colorbar for single class plots
```python
ax_single = fig.add_axes([0.15, 0.01, 0.5, 0.02])
plt.title("Probability")
_ = plt.colorbar(
    cm.ScalarMappable(norm=None, cmap=disp.surface_.cmap),
    cax=ax_single,
    orientation="horizontal",
)
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
ax_single = fig.add_axes([0.15, 0.01, 0.5, 0.02])
plt.title("Probability")
_ = plt.colorbar(
    cm.ScalarMappable(norm=None, cmap=disp.surface_.cmap),
    cax=ax_single,
    orientation="horizontal",
)
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
