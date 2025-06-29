# String Code Synthesis
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
## Synthesis Blocks
### notebooks/dataset2/classification/plot_digits_classification.ipynb
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
```
