# Embedding Code Synthesis
Query `Visualize projections of CCA vs PLS.`
## Script Variables
- np:<br>
>np is a python library that provides a large number of mathematical functions and data structures. It is a
- n:<br>
>n is an integer value that represents the number of samples in the dataset.
- connectivity:<br>
>The connectivity variable is a graph object that represents the connectivity between the nodes in the dataset. It is
- X:<br>
>X is a 2D array containing the input data points for the Swiss Roll dataset. Each row
- kneighbors_graph:<br>
>kneighbors_graph is a function from sklearn.neighbors module which is used to create a graph from a distance
- plt:<br>
>Plotting library in Python.
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

### notebooks/dataset2/clustering/plot_ward_structured_vs_unstructured.ipynb
CONTEXT:  We are defining k-Nearest Neighbors with 10 neighbors   COMMENT:
```python
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
```

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

### notebooks/dataset2/clustering/plot_coin_segmentation.ipynb
CONTEXT: Compute and visualize the resulting regions   COMMENT: To view individual segments as appear comment in plt.pause(0.5)
```python
plt.show()
```

## Code Concatenation
```python
import numpy as np
n = 500
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
import numpy as np
n = 500
plt.show()
```
