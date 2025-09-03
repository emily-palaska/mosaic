# Reverse Embedding Code Synthesis
Query `Run OPTICS clustering on noisy data.`
## Script Variables
- Xk:<br>
>Xk is a variable that stores the values of the space variable for each class. It is used
- X:<br>
>X is a matrix of 6 clusters, each cluster is a 2D array of points.
- color:<br>
>color is a list of strings that represent the colors to be used in the plot. The colors are
- enumerate:<br>
>enumerate() is a built-in function in Python that returns an enumerate object. It is used to iterate
- klass:<br>
>It is a variable that is used to iterate through the list of colors. The variable is used to
- ax3:<br>
>The variable ax3 is a plot of the reachability values of the data points in the clustering algorithm
- labels_050:<br>
>It is a variable that contains the cluster labels of the data points. The labels are assigned based on
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_optics.ipynb
CONTEXT:   Demo of OPTICS clustering algorithm  .. currentmodule:: sklearn  Finds core samples of high density and expands clusters from them. This
example uses data that is generated so that the clusters have different densities.  The :class:`~cluster.OPTICS` is first used with its Xi cluster
detection method, and then setting specific thresholds on the reachability, which corresponds to :class:`~cluster.DBSCAN`. We can see that the
different clusters of OPTICS's Xi method can be recovered with different choices of thresholds in DBSCAN.  COMMENT: DBSCAN at 0.5
```python
color = ["g.", "r.", "b.", "c."]
for klass, color in enumerate(color):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")
```

## Code Concatenation
```python
color = ["g.", "r.", "b.", "c."]
for klass, color in enumerate(color):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")
```
