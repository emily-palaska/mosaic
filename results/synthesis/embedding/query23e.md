# Embedding Code Synthesis
Query `Compress face images using cluster centers.`
## Script Variables
- plt:<br>
>plt is a Python library that provides a wide range of plotting functions and tools for data visualization. It
- _:<br>
>The variable _ is used to store the result of the KBinsDiscretizer.fit_transform() method
- encoder:<br>
>The variable encoder is a tool used to encode data into a binary format. It does this by dividing
- raccoon_face:<br>
>The variable raccoon_face is a 2D numpy array that represents the image of a raccoon
- compressed_raccoon_kmeans:<br>
>It is a variable that stores the compressed version of the raccoon_face dataset. The dataset is compressed
- fig:<br>
>fig is a variable that is used to store the figure object. It is used to create a plot
- ax:<br>
>ax is a matplotlib axis object that is used to display the histogram of the compressed pixel values of the
- n_bins:<br>
>n_bins is a variable that is used to specify the number of bins that will be used to discret
- KBinsDiscretizer:<br>
>KBinsDiscretizer is a class in scikit-learn that is used to discretize continuous
- gs:<br>
>gs is a 2D numpy array that is used to create a grid of subplots in a
- i:<br>
>The variable i is used to iterate through the list of classifiers. It is used to access the corresponding
- calibration_displays:<br>
>Calibration displays are used to evaluate the performance of a model in predicting the probability of a given outcome
- name:<br>
>The variable name is 'clf' which is a classifier. It is used to fit the training data
- CalibrationDisplay:<br>
>CalibrationDisplay is a class that helps visualize the calibration of a machine learning model. It is used
- X_test:<br>
>X_test is a matrix of size 100000 x 20 which is used to test the model
- clf_list:<br>
>It is a list of tuples, where each tuple contains a classifier and its name. The list is
- y_test:<br>
>y_test is a variable that is used to test the model. It is a test set that is
- enumerate:<br>
>The enumerate function is used to return the index of an iterable along with the value of the element.
- GridSpec:<br>
>GridSpec is a class in Matplotlib that allows you to create a grid of subplots in a
- clf:<br>
>clf is a classifier object that is used to fit the training data and predict the test data. It
- X_train:<br>
>X_train is a numpy array containing 100,000 samples of 20 features each.
- markers:<br>
>The markers are used to represent the different classes in the calibration curve. They are used to differentiate between
- y_train:<br>
>The variable y_train is a list of values that represent the actual values of the target variable in the
- display:<br>
>The variable display is a matplotlib object that is used to create calibration plots for machine learning models. It
- colors:<br>
>Colors are used to represent different classes in the histogram. The colors are chosen randomly from a list of
- ax_calibration_curve:<br>
>It is a matplotlib axis object that is used to plot calibration curves for each classifier in the ensemble.
- rng:<br>
>The variable rng is a random number generator. It is used to generate random numbers for the MiniBatch
- decomposition:<br>
>PCA is a dimensionality reduction technique that uses an orthogonal transformation to convert a set of observations of possibly
- cluster:<br>
>The variable cluster is a list of cluster centers. It is used to represent the cluster centers of the
- RandomState:<br>
>RandomState is a class that provides a random number generator for reproducible results. It is used to
- fetch_olivetti_faces:<br>
>It is a function that fetches the Olivetti faces dataset from the sklearn.datasets module.
- logging:<br>
>Logging is a mechanism that allows you to record information about the execution of your program. It is used
- center:<br>
>The variable center is a list of 256 values that represent the center of each bin in the histogram
- color:<br>
>The variable color is used to represent the color of the histogram bars. It is set to "tab
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_compare_calibration.ipynb
CONTEXT:  Calibration curves  Below, we train each of the four models with the small training dataset, then plot calibration curves (also known as
reliability diagrams) using predicted probabilities of the test dataset. Calibration curves are created by binning predicted probabilities, then
plotting the mean predicted probability in each bin against the observed frequency ('fraction of positives'). Below the calibration curve, we plot a
histogram showing the distribution of the predicted probabilities or more specifically, the number of samples in each predicted probability bin.
COMMENT:
```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")
```

### notebooks/dataset2/decomposition/plot_faces_decomposition.ipynb
CONTEXT:  Dataset preparation  Loading and preprocessing the Olivetti faces dataset.   COMMENT:
```python
import logging
import matplotlib.pyplot as plt
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
rng = RandomState(0)
```

### notebooks/dataset2/clustering/plot_face_compress.ipynb
CONTEXT: As previously stated, the uniform sampling strategy is not optimal. Notice for instance that the pixels mapped to the value 7 will encode a
rather small amount of information, whereas the mapped value 3 will represent a large amount of counts. We can instead use a clustering strategy such
as k-means to find a more optimal mapping.   COMMENT:
```python
_, ax = plt.subplots()
ax.hist(raccoon_face.ravel(), bins=256)
color = "tab:orange"
for center in center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()[1] + 100, f"{center:.1f}", color=color)
```

### notebooks/dataset2/clustering/plot_face_compress.ipynb
CONTEXT: As previously stated, the uniform sampling strategy is not optimal. Notice for instance that the pixels mapped to the value 7 will encode a
rather small amount of information, whereas the mapped value 3 will represent a large amount of counts. We can instead use a clustering strategy such
as k-means to find a more optimal mapping.   COMMENT:
```python
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Rendering of the image")
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")
ax[1].set_ylabel("Number of pixels")
ax[1].set_title("Distribution of the pixel values")
_ = fig.suptitle("Raccoon raccoon_face compressed using 3 bits and a K-means strategy")
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")
ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")
import logging
import matplotlib.pyplot as plt
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
rng = RandomState(0)
_, ax = plt.subplots()
ax.hist(raccoon_face.ravel(), bins=256)
color = "tab:orange"
for center in center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()[1] + 100, f"{center:.1f}", color=color)
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Rendering of the image")
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")
ax[1].set_ylabel("Number of pixels")
ax[1].set_title("Distribution of the pixel values")
_ = fig.suptitle("Raccoon raccoon_face compressed using 3 bits and a K-means strategy")
```
