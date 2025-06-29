# Reverse Embedding Code Synthesis
Query `Create classifiers with names Regression, SVM, Tree, AdaBoost and Bayes classifiers. Compare them and plot them.`
## Script Variables
- make_pipeline:<br>
>make_pipeline() is a function in scikit-learn that allows us to create a pipeline of machine
- classifiers:<br>
>The variable classifiers are used to classify the input data into different classes. They are used to identify the
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
- plt:<br>
>plt is a module in Python that is used for plotting and graphing. It is a part of
- ax:<br>
>ax is a scatter plot object that is used to plot the training data points on the scatter plot.
- ds_cnt:<br>
>It is a counter that keeps track of the number of datasets in the list of datasets. It is
- datasets:<br>
>The variable datasets are the datasets used to train the machine learning models. They are used to predict the
- cm_bright:<br>
>cm_bright is a colormap that is used to color the scatter plot. It is a color map
- len:<br>
>len is a function that returns the length of an object. In this case, it is used to
- i:<br>
>The variable i is a counter that is used to keep track of the number of plots that have been
- ListedColormap:<br>
>It is a colormap that is used to represent the color of the points in the scatter plot. The
## Synthesis Blocks
### notebooks/dataset2/classification/plot_classifier_comparison.ipynb
CONTEXT:   Classifier comparison  A comparison of several classifiers in scikit-learn on synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these
examples does not necessarily carry over to real datasets.  Particularly in high-dimensional spaces, data can more easily be separated linearly and
the simplicity of classifiers such as naive Bayes and linear SVMs might lead to better generalization than is achieved by other classifiers.  The
plots show training points in solid colors and testing points semi-transparent. The lower right shows the classification accuracy on the test set.
COMMENT: just plot the dataset first
```python
cm_bright = plt.cm_bright.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
if ds_cnt == 0:
    ax.set_title("Input data")
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
cm_bright = plt.cm_bright.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
if ds_cnt == 0:
    ax.set_title("Input data")
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
