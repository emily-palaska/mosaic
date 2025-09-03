# Reverse Embedding Code Synthesis
Query `Compare cluster shapes using different methods.`
## Script Variables
- KMeans:<br>
>KMeans is a clustering algorithm that uses an iterative approach to partition n observations into k clusters, where
- BisectingKMeans:<br>
>BisectingKMeans is a clustering algorithm that uses a divide and conquer approach to find the
- clustering_algorithms:<br>
>It is a dictionary that contains the name of the clustering algorithm and the corresponding class that implements it.
## Synthesis Blocks
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
clustering_algorithms = {
    "Bisecting K-Means": BisectingKMeans,
    "K-Means": KMeans,
}
```
