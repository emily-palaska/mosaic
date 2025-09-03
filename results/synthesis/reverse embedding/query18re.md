# Reverse Embedding Code Synthesis
Query `Compare structured vs unstructured Ward clustering.`
## Script Variables
- time:<br>
>The variable time is a built-in function in Python that returns the current time in seconds since the epoch
- X_test:<br>
>X_test is a numpy array containing the test data. It is used to plot the test data points
- all_models:<br>
>It is a dictionary that contains the Gradient Boosting Regressor models with different alpha values.
- coverage_fraction:<br>
>The coverage_fraction variable is used to calculate the coverage of the prediction of the model. It is calculated
- y_test:<br>
>It is a test set of the data used to evaluate the model's performance. It is used to
- i:<br>
>The variable i is used to index the important words in the bicluster. It is used to
- cocluster:<br>
>Cocluster is a variable that is used to identify the documents that are not part of the cluster
- bicluster_ncut:<br>
>It is a function that calculates the number of clusters in the bicluster.
- np:<br>
>np is a library in python that is used for scientific computing. It provides a large set of mathematical
- X:<br>
>X is a matrix of size n x m where n is the number of documents and m is the
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_ward_structured_vs_unstructured.ipynb
CONTEXT:   Hierarchical clustering: structured vs unstructured ward  Example builds a swiss roll dataset and runs hierarchical clustering on their
position.  For more information, see `hierarchical_clustering`.  In a first step, the hierarchical clustering is performed without connectivity
constraints on the structure and is solely based on distance, whereas in a second step the clustering is restricted to the k-Nearest Neighbors graph:
it's a hierarchical clustering with structure prior.  Some of the clusters learned without connectivity constraints do not respect the structure of
the swiss roll and extend across different folds of the manifolds. On the opposite, when opposing connectivity constraints, the clusters form a nice
parcellation of the swiss roll.  COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import time as time
```

### notebooks/dataset2/biclustering/plot_bicluster_newsgroups.ipynb
CONTEXT:   Biclustering documents with the Spectral Co-clustering algorithm  This example demonstrates the Spectral Co-clustering algorithm on the
twenty newsgroups dataset. The 'comp.os.ms-windows.misc' category is excluded because it contains many posts containing nothing but data.  The TF-IDF
vectorized posts form a word frequency matrix, which is then biclustered using Dhillon's Spectral Co-Clustering algorithm. The resulting document-word
biclusters indicate subsets words used more often in those subsets documents.  For a few of the best biclusters, its most common document categories
and its ten most important words get printed. The best biclusters are determined by their normalized cut. The best words are determined by comparing
their sums inside and outside the bicluster.  For comparison, the documents are also clustered using MiniBatchKMeans. The document clusters derived
from the biclusters achieve a better V-measure than clusters found by MiniBatchKMeans.  COMMENT: Note: the following is identical to X[rows[:,
np.newaxis], cols].sum() but much faster in scipy <= 0.16
```python
def bicluster_ncut(i):    rows, cols = cocluster.get_indices(i)    if not (np.any(rows) and np.any(cols)):        import sys        return sys.float_info.max    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]    weight = X[rows][:, cols].sum()    cut = X[row_complement][:, cols].sum() + X[rows][:, col_complement].sum()    return cut / weight
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: On the training set the calibration is very close to the expected coverage value for a 90% confidence interval.   COMMENT:
```python
coverage_fraction(
    y_test, all_models["q 0.05"].predict(X_test), all_models["q 0.95"].predict(X_test)
)
```

## Code Concatenation
```python
import time as time
def bicluster_ncut(i):    rows, cols = cocluster.get_indices(i)    if not (np.any(rows) and np.any(cols)):        import sys        return sys.float_info.max    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]    weight = X[rows][:, cols].sum()    cut = X[row_complement][:, cols].sum() + X[rows][:, col_complement].sum()    return cut / weight
coverage_fraction(
    y_test, all_models["q 0.05"].predict(X_test), all_models["q 0.95"].predict(X_test)
)
```
