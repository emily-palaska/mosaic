# Reverse Embedding Code Synthesis
Query `How to perform cross_decomposition`
## Script Variables
- np:<br>
>It is a python library that provides a wide range of mathematical functions and tools for scientific computing. It
- n:<br>
>n is the number of samples in the dataset.
- clf:<br>
>clf is a classifier object which is used to predict the class of the test data.
- svm:<br>
>svm is a classification algorithm that uses a support vector machine to classify data. The gamma parameter controls the
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_compare_cross_decomposition.ipynb
CONTEXT:   Compare cross decomposition methods  Simple usage of various cross decomposition algorithms:  - PLSCanonical - PLSRegression, with
multivariate response, a.k.a. PLS2 - PLSRegression, with univariate response, a.k.a. PLS1 - CCA  Given 2 multivariate covarying two-dimensional
datasets, X, and Y, PLS extracts the 'directions of covariance', i.e. the components of each datasets that explain the most shared variance between
both datasets. This is apparent on the **scatterplot matrix** display: components 1 in dataset X and dataset Y are maximally correlated (points lie
around the first diagonal). This is also true for components 2 in both dataset, however, the correlation across datasets for different components is
weak: the point cloud is very spherical.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import numpy as np
n = 500
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
import numpy as np
n = 500
clf = svm.SVC(gamma=0.001)
```
