# Exhaustive Code Synthesis
Query `Graph operations`
## Script Variables
- image:<br>
>The variable image is a 2D numpy array that represents the input image. It is used to
- digits:<br>
>It is a 2D array of size 28x28, which is the image of the
- cm:<br>
>cm is a 2D array of integers, where each row represents the number of times a given
- zip:<br>
>The zip() function is a built-in function in Python that takes an iterable (a sequence, list
- _:<br>
>The variable _ is a tuple that contains the axes of the subplots. The axes are used to
- axes:<br>
>It is a variable that is used to create a grid of subplots. It is used to create
- plt:<br>
>plt is a python library that is used to create plots in python. It is a python library that
- ax:<br>
>ax is a variable that is used to store the axes of the subplots. It is used to
- datasets:<br>
>The variable datasets is a set of data that is used to train a machine learning model. It contains
- label:<br>
>The variable label is a list of 64x64 pixel images of digits. The images are in
- metrics:<br>
>The variable metrics is a function that calculates the classification report for a given classifier. It takes two arguments
- predicted:<br>
>The variable predicted is a variable that is used to predict the output of the model. It is a
- y_test:<br>
>y_test is a numpy array that contains the actual values of the test data. It is used to
- disp:<br>
>disp is a confusion matrix that is used to compare the predicted values with the actual values.
- print:<br>
>print() is a function that prints a string to the console. In this case, it is used
- X:<br>
>X is a matrix of size n x 4, where n is the number of samples. Each
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
- X_train_r:<br>
>X_train_r is a matrix of reduced dimensionality that contains the principal components of the training data.
- Y_train_r:<br>
>Y_train_r is a matrix of dimension 1000 x 2 that contains the results of the
- Y:<br>
>Y is a matrix of size n x q where n is the number of samples and q is the
- X_test_r:<br>
>X_test_r is a matrix of the test data, which is used to calculate the correlation between the
- Y_test_r:<br>
>Y_test_r is the transformed version of the test data, which is the output of the PLSC
## Synthesis Blocks
### notebooks/plot_digits_classification.ipynb
CONTEXT:   Recognizing hand-written digits  This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.
COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause Standard scientific Python imports
```python
import matplotlib.pyplot as plt
```

### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:  Canonical (symmetric) PLS   Transform data   COMMENT:
```python
import matplotlib.pyplot as plt
```

### notebooks/plot_digits_classification.ipynb
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

### notebooks/plot_digits_classification.ipynb
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
