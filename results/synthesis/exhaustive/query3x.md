# Exhaustive Code Synthesis
Query `Plot predicted probabilities vs true outcomes.`
## Script Variables
- train_samples:<br>
>It is a variable that stores the number of samples to be used for training the model. This number
- plt:<br>
>plt is a Python library that provides a wide range of visualization tools for data analysis and visualization. It
- _:<br>
>The variable _ is used to store the legend object in the plot. It is used to label the
- x:<br>
>It is a variable that is used to iterate over a list of values. The list contains 11
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_compare_calibration.ipynb
CONTEXT:   Comparison of Calibration of Classifiers  Well calibrated classifiers are probabilistic classifiers for which the output of
:term:`predict_proba` can be directly interpreted as a confidence level. For instance, a well calibrated (binary) classifier should classify the
samples such that for the samples to which it gave a :term:`predict_proba` value close to 0.8, approximately 80% actually belong to the positive
class.  In this example we will compare the calibration of four different models: `Logistic_regression`, `gaussian_naive_bayes`, `Random Forest
Classifier <forest>` and `Linear SVM <svm_classification>`. Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause   COMMENT:
Samples used for training the models
```python
train_samples = 100
```

### notebooks/dataset2/calibration/plot_calibration_multiclass.ipynb
CONTEXT:  Compare probabilities Below we plot a 2-simplex with arrows showing the change in predicted probabilities of the test samples.   COMMENT:
Add grid
```python
plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)
plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")
```

## Code Concatenation
```python
train_samples = 100
plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)
plt.title("Change of predicted probabilities on test samples after sigmoid calibration")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
_ = plt.legend(loc="best")
```
