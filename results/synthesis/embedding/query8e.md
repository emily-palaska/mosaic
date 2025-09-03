# Embedding Code Synthesis
Query `Explore cluster over‑segmentation effects.`
## Script Variables
- n_clusters:<br>
>n_clusters is a variable that is used to determine the number of clusters that will be created. It
- plt:<br>
>plt is a module that provides a number of functions for creating and manipulating plots.
- float:<br>
>The variable float is a floating-point number that is used to represent decimal numbers in Python. It is
- rescaled_coins:<br>
>The rescaled_coins variable is a 2D numpy array that represents the input image. It
- label:<br>
>The variable label is a 2D numpy array of size (100, 100) which represents
- l:<br>
>l is a variable that is used to iterate over the range of n_clusters.
- range:<br>
>The variable range is from 0 to 1. The range of values for each variable is determined
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_coin_ward_segmentation.ipynb
CONTEXT:  Plot the results on an image  Agglomerative clustering is able to segment each coin however, we have had to use a ``n_cluster`` larger than
the number of coins because the segmentation is finding a large in the background.   COMMENT:
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.imshow(rescaled_coins, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(
        label == l,
        colors=[
            plt.cm.nipy_spectral(l / float(n_clusters)),
        ],
    )
plt.axis("off")
plt.show()
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.imshow(rescaled_coins, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(
        label == l,
        colors=[
            plt.cm.nipy_spectral(l / float(n_clusters)),
        ],
    )
plt.axis("off")
plt.show()
```
