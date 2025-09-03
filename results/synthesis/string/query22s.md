# String Code Synthesis
Query `Plot clustered data from spectral coclustering.`
## Script Variables
- np:<br>
>The variable np is a Python package that provides a large collection of mathematical functions and data structures. It
- img:<br>
>The variable img is a 2D numpy array that represents an image. It is used to extract
- label_im:<br>
>Label_im is a 2D numpy array that represents the labels of each pixel in the image.
- mask:<br>
>The mask variable is used to indicate which pixels in the image should be considered for the clustering algorithm.
- axs:<br>
>axs is a variable that is used to create a figure with two subplots. The first subplot
- fig:<br>
>fig is a figure object that is used to display the image and label image. It is created using
- labels:<br>
>labels
- graph:<br>
>The variable graph is a function that takes an image and a mask as input and returns a graph representation
- spectral_clustering:<br>
>Spectral clustering is a clustering algorithm that uses the eigenvectors of the graph Laplacian to
- plt:<br>
>plt is a python library that is used to create plots in python. It is a high-level library
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_segmentation_toy.ipynb
CONTEXT: Here we perform spectral clustering using the arpack solver since amg is numerically unstable on this example. We then plot the results.
COMMENT:
```python
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering
labels = spectral_clustering(graph, n_clusters=4, eigen_solver="arpack")
label_im = np.full(mask.shape, -1.0)
label_im[mask] = labels
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].matshow(img)
axs[1].matshow(label_im)
plt.show()
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
from sklearn.cluster import spectral_clustering
labels = spectral_clustering(graph, n_clusters=4, eigen_solver="arpack")
label_im = np.full(mask.shape, -1.0)
label_im[mask] = labels
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axs[0].matshow(img)
axs[1].matshow(label_im)
plt.show()
```
