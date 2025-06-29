# Exhaustive Code Synthesis
Query `Graph operations`
## Script Variables
- plt:<br>
>plt is a module that provides a number of command-line interfaces for plotting in Python. It is a
- digits:<br>
>It is a variable that is used to store the digits of the image. It is a 2
- _:<br>
>The variable _ is a placeholder for the value of the variable digits, which is a dictionary containing the
- zip:<br>
>It is a function that takes two arguments, a list of lists and a list of strings. It
- ax:<br>
>The variable ax is a subplot object that is used to display the images in the script. It is
- datasets:<br>
>The dataset is a collection of 1797 images of handwritten digits, each image is a 8
- cm:<br>
>cm is a confusion matrix that is used to compare the actual values of the target variable with the predicted
- image:<br>
>Image is a 2D array of size 8x8. It contains the pixels of the
- axes:<br>
>The variable axes is a tuple containing the axes of the subplots. It is used to create a
- label:<br>
>The variable label is a dataset that contains 1797 images of digits from 0 to 9
- np:<br>
>The np variable is a Python package that provides a large collection of mathematical functions and data structures. It
- Y_test_r:<br>
>Y_test_r is a matrix of the same size as X_test, where each row is the result
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
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Canonical (symmetric) PLS   Transform data   COMMENT:
```python
import matplotlib.pyplot as plt
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT:   Recognizing hand-written digits  This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.
COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause Standard scientific Python imports
```python
import matplotlib.pyplot as plt
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT:  Digits dataset  The digits dataset consists of 8x8 pixel images of digits. The ``images`` attribute of the dataset stores 8x8 arrays of
grayscale values for each image. We will use these arrays to visualize the first 4 images. The ``target`` attribute of the dataset stores the digit
each image represents and this is included in the title of the 4 plots below.  Note: if we were working from image files (e.g., 'png' files), we would
load them using :func:`matplotlib.pyplot.imread`.   COMMENT:
```python
digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT: We can also plot a `confusion matrix <confusion_matrix>` of the true digit values and the predicted digit values.   COMMENT:
```python
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Scatter plot of scores   COMMENT: On diagonal plot X vs Y scores on each components
```python
plt.subplot(222)
plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train", marker="*", s=50)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test", marker="*", s=50)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title(
    "X comp. 1 vs X comp. 2 (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1]
)
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.subplot(223)
plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train", marker="*", s=50)
plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test", marker="*", s=50)
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.title(
    "Y comp. 1 vs Y comp. 2 , (test corr = %.2f)"
    % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1]
)
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.show()
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
plt.subplot(222)
plt.scatter(X_train_r[:, 0], X_train_r[:, 1], label="train", marker="*", s=50)
plt.scatter(X_test_r[:, 0], X_test_r[:, 1], label="test", marker="*", s=50)
plt.xlabel("X comp. 1")
plt.ylabel("X comp. 2")
plt.title(
    "X comp. 1 vs X comp. 2 (test corr = %.2f)"
    % np.corrcoef(X_test_r[:, 0], X_test_r[:, 1])[0, 1]
)
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.subplot(223)
plt.scatter(Y_train_r[:, 0], Y_train_r[:, 1], label="train", marker="*", s=50)
plt.scatter(Y_test_r[:, 0], Y_test_r[:, 1], label="test", marker="*", s=50)
plt.xlabel("Y comp. 1")
plt.ylabel("Y comp. 2")
plt.title(
    "Y comp. 1 vs Y comp. 2 , (test corr = %.2f)"
    % np.corrcoef(Y_test_r[:, 0], Y_test_r[:, 1])[0, 1]
)
plt.legend(loc="best")
plt.xticks(())
plt.yticks(())
plt.show()
```
