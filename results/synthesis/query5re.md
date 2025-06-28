# Reverse Embedding Code Synthesis
Query `PCA algorithm`
## Script Variables
- pca:<br>
>pca is a PCA object which is used to perform Principal Component Analysis. PCA is a dimensionality
- pcr:<br>
>The variable pcr is a function that calculates the r-squared value of a given dataset. It
- clf:<br>
>clf is a classifier object which is used to predict the class of the test data.
- svm:<br>
>svm is a classification algorithm that uses a support vector machine to classify data. The gamma parameter controls the
- np:<br>
>The variable np is a Python package that provides a large number of functions for working with arrays and matrices
- make_data:<br>
>make_data is a function that generates synthetic data for the purpose of testing the performance of a classifier.
- cov_class_2:<br>
>It is a 2x2 matrix that represents the covariance between the two classes. It is used
- cov_class_1:<br>
>cov_class_1 is a numpy array that represents the covariance matrix of the data points in the dataset
- y:<br>
>The variable y is a 2D array of size (n_samples, 1) where n
- X:<br>
>X is a 2D array of shape (n_samples, n_features) representing the data.
## Synthesis Blocks
### notebooks/plot_lda_qda.ipynb
CONTEXT:  Data generation  First, we define a function to generate synthetic data. It creates two blobs centered at `(0, 0)` and `(1, 1)`. Each blob
is assigned a specific class. The dispersion of the blob is controlled by the parameters `cov_class_1` and `cov_class_2`, that are the covariance
matrices used when generating the samples from the Gaussian distributions.   COMMENT:
```python
def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):    rng = np.random.RandomState(seed)    X = np.concatenate(        [            rng.randn(n_samples, n_features) @ cov_class_1,            rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1]),        ]    )    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])    return X, y
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/plot_digits_classification.ipynb
CONTEXT:  Classification  To apply a classifier on this data, we need to flatten the images, turning each 2-D array of grayscale values from shape
``(8, 8)`` into shape ``(64,)``. Subsequently, the entire dataset will be of shape ``(n_samples, n_features)``, where ``n_samples`` is the number of
images and ``n_features`` is the total number of pixels in each image.  We can then split the data into train and test subsets and fit a support
vector classifier on the train samples. The fitted classifier can subsequently be used to predict the value of the digit for the samples in the test
subset.   COMMENT: Create a classifier: a support vector classifier
```python
clf = svm.SVC(gamma=0.001)
```

## Code Concatenation
```python
def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):    rng = np.random.RandomState(seed)    X = np.concatenate(        [            rng.randn(n_samples, n_features) @ cov_class_1,            rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1]),        ]    )    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])    return X, y
pca = pcr.named_steps["pca"]
clf = svm.SVC(gamma=0.001)
```
