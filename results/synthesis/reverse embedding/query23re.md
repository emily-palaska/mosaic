# Reverse Embedding Code Synthesis
Query `Compress face images using cluster centers.`
## Script Variables
- plt:<br>
>plt is a Python library that provides a wide range of plotting functions and tools for data visualization. It
- _:<br>
>The variable _ is used to store the result of the KBinsDiscretizer.fit_transform() method
- encoder:<br>
>The variable encoder is a tool used to encode data into a binary format. It does this by dividing
- raccoon_face:<br>
>The variable raccoon_face is a 2D numpy array that represents the image of a raccoon
- compressed_raccoon_kmeans:<br>
>It is a variable that stores the compressed version of the raccoon_face dataset. The dataset is compressed
- fig:<br>
>The variable fig is a matplotlib figure object that represents the entire figure. It is used to create and
- ax:<br>
>ax is a matplotlib axis object that is used to display the histogram of the compressed pixel values of the
- n_bins:<br>
>n_bins is a variable that is used to specify the number of bins that will be used to discret
- KBinsDiscretizer:<br>
>KBinsDiscretizer is a class in scikit-learn that is used to discretize continuous
## Synthesis Blocks
### notebooks/dataset2/clustering/plot_face_compress.ipynb
CONTEXT: As previously stated, the uniform sampling strategy is not optimal. Notice for instance that the pixels mapped to the value 7 will encode a
rather small amount of information, whereas the mapped value 3 will represent a large amount of counts. We can instead use a clustering strategy such
as k-means to find a more optimal mapping.   COMMENT:
```python
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Rendering of the image")
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")
ax[1].set_ylabel("Number of pixels")
ax[1].set_title("Distribution of the pixel values")
_ = fig.suptitle("Raccoon raccoon_face compressed using 3 bits and a K-means strategy")
```

## Code Concatenation
```python
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Rendering of the image")
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("Pixel value")
ax[1].set_ylabel("Number of pixels")
ax[1].set_title("Distribution of the pixel values")
_ = fig.suptitle("Raccoon raccoon_face compressed using 3 bits and a K-means strategy")
```
