# Embedding Code Synthesis
Query `Analyze cluster stability across multiple runs.`
## Script Variables
- n_runs:<br>
>The variable n_runs is used to specify the number of times the script will be executed. This is
- n_init_range:<br>
>n_init_range is a list of integers that represents the number of times the k-means algorithm is
- np:<br>
>np is a python library that provides a large collection of mathematical functions for scientific computing. It is used
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_kmeans_stability_low_dim_dense.ipynb
CONTEXT:   Empirical evaluation of the impact of k-means initialization  Evaluate the ability of k-means initializations strategies to make the
algorithm convergence robust, as measured by the relative standard deviation of the inertia of the clustering (i.e. the sum of squared distances to
the nearest cluster center).  The first plot shows the best inertia reached for each combination of the model (``KMeans`` or ``MiniBatchKMeans``), and
the init method (``init="random"`` or ``init="k-means++"``) for increasing values of the ``n_init`` parameter that controls the number of
initializations.  The second plot demonstrates one single run of the ``MiniBatchKMeans`` estimator using a ``init="random"`` and ``n_init=1``. This
run leads to a bad convergence (local optimum), with estimated centers stuck between ground truth clusters.  The dataset used for evaluation is a 2D
grid of isotropic Gaussian clusters widely spaced.  COMMENT: Number of run (with randomly generated dataset) for each strategy so as to be able to
compute an estimate of the standard deviation
```python
n_runs = 5
```

### notebooks/dataset2/clustering/plot_kmeans_stability_low_dim_dense.ipynb
CONTEXT:   Empirical evaluation of the impact of k-means initialization  Evaluate the ability of k-means initializations strategies to make the
algorithm convergence robust, as measured by the relative standard deviation of the inertia of the clustering (i.e. the sum of squared distances to
the nearest cluster center).  The first plot shows the best inertia reached for each combination of the model (``KMeans`` or ``MiniBatchKMeans``), and
the init method (``init="random"`` or ``init="k-means++"``) for increasing values of the ``n_init`` parameter that controls the number of
initializations.  The second plot demonstrates one single run of the ``MiniBatchKMeans`` estimator using a ``init="random"`` and ``n_init=1``. This
run leads to a bad convergence (local optimum), with estimated centers stuck between ground truth clusters.  The dataset used for evaluation is a 2D
grid of isotropic Gaussian clusters widely spaced.  COMMENT: k-means models can do several random inits so as to be able to trade CPU time for
convergence robustness
```python
n_init_range = np.array([1, 5, 10, 15, 20])
```

## Code Concatenation
```python
n_runs = 5
n_init_range = np.array([1, 5, 10, 15, 20])
```
