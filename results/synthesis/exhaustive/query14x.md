# Exhaustive Code Synthesis
Query `Plot calibration curves for multiple classifiers.`
## Script Variables
- x_max:<br>
>x_max is the maximum value of x that is used to plot the decision boundary. It is used
- y:<br>
>The variable y is a numpy array that contains the target values for the given dataset. It is used
- plt:<br>
>plt is a Python library that is used to create plots in Python. It is a powerful and flexible
- y_min:<br>
>The variable y_min is the minimum value of the y axis. It is used to set the limits
- disp:<br>
>disp is a DecisionBoundaryDisplay object that is used to display the decision boundary of the bdt model
- x_min:<br>
>The variable x_min is the minimum value of the x-axis. It is used to set the limits
- y_max:<br>
>The variable y_max is the maximum value of the y variable in the dataset. It is used to
- bdt:<br>
>The variable bdt is a Decision Tree Classifier model that is used to predict the class of a given
- X:<br>
>X is a matrix of size (n_samples, n_features) where n_samples is the number of
- ax:<br>
>The variable ax is a subplot object that is used to display the decision boundary of the boosted decision tree
- DecisionBoundaryDisplay:<br>
>DecisionBoundaryDisplay is a class that is used to visualize the decision boundary of a decision tree. It
- LinearSVC:<br>
>LinearSVC is a class from sklearn.svm module which is used for linear support vector classification
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_calibration_curve.ipynb
CONTEXT: Uncalibrated :class:`~sklearn.naive_bayes.GaussianNB` is poorly calibrated because of the redundant features which violate the assumption of
feature-independence and result in an overly confident classifier, which is indicated by the typical transposed-sigmoid curve. Calibration of the
probabilities of :class:`~sklearn.naive_bayes.GaussianNB` with `isotonic` can fix this issue as can be seen from the nearly diagonal calibration
curve. `Sigmoid regression <sigmoid_regressor>` also improves calibration slightly, albeit not as strongly as the non-parametric isotonic regression.
This can be attributed to the fact that we have plenty of calibration data such that the greater flexibility of the non-parametric model can be
exploited.  Below we will make a quantitative analysis considering several classification metrics: `brier_score_loss`, `log_loss`, `precision, recall,
F1 score <precision_recall_f_measure_metrics>` and `ROC AUC <roc_metrics>`.   COMMENT:
```python
import numpy as np
from sklearn.svm import LinearSVC
```

### notebooks/dataset2/ensemble_methods/plot_adaboost_twoclass.ipynb
CONTEXT:   Two-class AdaBoost  This example fits an AdaBoosted decision stump on a non-linearly separable classification dataset composed of two
"Gaussian quantiles" clusters (see :func:`sklearn.datasets.make_gaussian_quantiles`) and plots the decision boundary and decision scores. The
distributions of decision scores are shown separately for samples of class A and B. The predicted class label for each sample is determined by the
sign of the decision score. Samples with decision scores greater than zero are classified as B, and are otherwise classified as A. The magnitude of a
decision score determines the degree of likeness with the predicted class label. Additionally, a new dataset could be constructed containing a desired
purity of class B, for example, by only selecting samples with a decision score above some value.  COMMENT: Plot the decision boundaries
```python
ax = plt.subplot(121)
disp = DecisionBoundaryDisplay.from_estimator(
    bdt,
    X,
    cmap=plt.cm.Paired,
    response_method="predict",
    ax=ax,
    xlabel="x",
    ylabel="y",
)
x_min, x_max = disp.xx0.min(), disp.xx0.max()
y_min, y_max = disp.xx1.min(), disp.xx1.max()
plt.axis("tight")
```

## Code Concatenation
```python
import numpy as np
from sklearn.svm import LinearSVC
ax = plt.subplot(121)
disp = DecisionBoundaryDisplay.from_estimator(
    bdt,
    X,
    cmap=plt.cm.Paired,
    response_method="predict",
    ax=ax,
    xlabel="x",
    ylabel="y",
)
x_min, x_max = disp.xx0.min(), disp.xx0.max()
y_min, y_max = disp.xx1.min(), disp.xx1.max()
plt.axis("tight")
```
