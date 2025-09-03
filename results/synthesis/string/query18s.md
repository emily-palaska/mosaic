# String Code Synthesis
Query `Compare structured vs unstructured Ward clustering.`
## Script Variables
- time:<br>
>The variable time is a built-in function in Python that returns the current time in seconds since the epoch
- KMeans:<br>
>KMeans is a clustering algorithm that uses an iterative approach to partition n observations into k clusters, where
- BisectingKMeans:<br>
>BisectingKMeans is a clustering algorithm that uses a divide and conquer approach to find the
- clustering_algorithms:<br>
>It is a dictionary that contains the name of the clustering algorithm and the corresponding class that implements it.
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

### notebooks/dataset2/clustering/plot_bisect_kmeans.ipynb
CONTEXT:   Bisecting K-Means and Regular K-Means Performance Comparison  This example shows differences between Regular K-Means algorithm and
Bisecting K-Means.  While K-Means clusterings are different when increasing n_clusters, Bisecting K-Means clustering builds on top of the previous
ones. As a result, it tends to create clusters that have a more regular large-scale structure. This difference can be visually observed: for all
numbers of clusters, there is a dividing line cutting the overall data cloud in two for BisectingKMeans, which is not present for regular K-Means.
COMMENT: Algorithms to compare
```python
clustering_algorithms = {
    "Bisecting K-Means": BisectingKMeans,
    "K-Means": KMeans,
}
```

## Code Concatenation
```python
import time as time
clustering_algorithms = {
    "Bisecting K-Means": BisectingKMeans,
    "K-Means": KMeans,
}
```
