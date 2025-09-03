# Exhaustive Code Synthesis
Query `Compare PCR and PLS regression results.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to perform Principal Component Analysis on the data. It is
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- var:<br>
>var is a variable that is used to multiply the value of comp by the value of var.
- comp:<br>
>The variable comp is a list of tuples, where each tuple represents a component of the principal components of
- enumerate:<br>
>The enumerate() function returns a list of tuples where the first element of each tuple is the index of
- cov:<br>
>The variable cov is a 2x2 matrix that represents the covariance between the two features in the
- X:<br>
>X is a numpy array containing the data points for the first PCA component. It is a 2
- PCA:<br>
>PCA stands for Principal Component Analysis. It is a dimensionality reduction technique that transforms a set of correlated
- rng:<br>
>The variable rng is a random number generator that is used to generate random numbers for the train-test split
- plt:<br>
>plt is a module in python which is used for plotting graphs. It is a part of the matplotlib
- i:<br>
>i is a variable that represents the number of components to be used in the PCA algorithm. It is
- np:<br>
>Numpy is a Python library that provides a multidimensional array object, which is a generalization of
- n_samples:<br>
>n_samples is the number of samples in the dataset. It is used to generate random noise in the
- zip:<br>
>The zip() function is used to create an iterator that aggregates elements from two or more iterables.
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  The data  We start by creating a simple dataset with two features. Before we even dive into PCR and PLS, we fit a PCA estimator to display
the two principal components of this dataset, i.e. the two directions that explain the most variance in the data.   COMMENT: scale component by its
variance explanation power
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = PCA(n_components=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var

    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)
plt.legend()
plt.show()
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
pca = PCA(n_components=2).fit(X)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="samples")
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var

    plt.plot(
        [0, comp[0]],
        [0, comp[1]],
        label=f"Component {i}",
        linewidth=5,
        color=f"C{i + 2}",
    )
plt.gca().set(
    aspect="equal",
    title="2-dimensional dataset with principal components",
    xlabel="first feature",
    ylabel="second feature",
)
plt.legend()
plt.show()
pca = pcr.named_steps["pca"]
```
