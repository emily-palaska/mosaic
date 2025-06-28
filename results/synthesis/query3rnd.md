# Random Code Synthesis
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
## Synthesis Blocks
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

## Code Concatenation
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
```
