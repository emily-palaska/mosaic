# Reverse Embedding Code Synthesis
Query `Simple PCA algorithm.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to reduce the dimensionality of the data. It does
- pcr:<br>
>The variable pcr is a pipeline that contains a standard scaler, a PCA component, and a linear
- cov_class_1:<br>
>The variable cov_class_1 is a 2x2 matrix that represents the covariance of the data
- y:<br>
>The variable y is used to represent the class labels of the data points in the dataset. It is
- X:<br>
>X is a numpy array of shape (n_samples, n_features) containing the data points. The
- np:<br>
>The variable np is a Python library that provides a large number of functions for working with arrays and matrices
- make_data:<br>
>The make_data function is used to generate synthetic data for the purpose of testing the performance of a machine
- clf:<br>
>It is a classifier that is used to predict the class of a given data point.
- svm:<br>
>svm is a short form of support vector machine. It is a machine learning algorithm that is used for
## Synthesis Blocks
### notebooks/dataset2/classification/plot_lda_qda.ipynb
CONTEXT:  Data generation  First, we define a function to generate synthetic data. It creates two blobs centered at `(0, 0)` and `(1, 1)`. Each blob
is assigned a specific class. The dispersion of the blob is controlled by the parameters `cov_class_1` and `cov_class_2`, that are the covariance
matrices used when generating the samples from the Gaussian distributions.   COMMENT:
```python
def make_data(n_samples, n_features, cov_class_1, cov_class_1, seed=0):    rng = np.random.RandomState(seed)    X = np.concatenate(        [            rng.randn(n_samples, n_features) @ cov_class_1,            rng.randn(n_samples, n_features) @ cov_class_1 + np.array([1, 1]),        ]    )    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])    return X, y
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/dataset2/classification/plot_digits_classification.ipynb
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
def make_data(n_samples, n_features, cov_class_1, cov_class_1, seed=0):    rng = np.random.RandomState(seed)    X = np.concatenate(        [            rng.randn(n_samples, n_features) @ cov_class_1,            rng.randn(n_samples, n_features) @ cov_class_1 + np.array([1, 1]),        ]    )    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])    return X, y
pca = pcr.named_steps["pca"]
clf = svm.SVC(gamma=0.001)
```
