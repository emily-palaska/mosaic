# String Code Synthesis
Query `Run GMM on sinusoidal synthetic data.`
## Script Variables
- time:<br>
>The variable time is a numpy array that contains 2000 values ranging from 0 to 8
- np:<br>
>It is a Python module that provides a number of useful tools for scientific computing, with a strong focus
- s1:<br>
>The variable s1 is a sine function of time. It is used to generate a sine wave that
- sample_weight:<br>
>Sample weight is a variable that is used to indicate the importance of each sample in the dataset. It
- y:<br>
>The variable y is the target variable of the dataset. It is the variable that we want to predict
- X:<br>
>X is a 2D array containing the data points of the dataset. It is a random sample
- n_samples:<br>
>n_samples is a variable that is used to specify the number of samples to generate for the dataset.
- centers:<br>
>Centers is a list of tuples that represent the coordinates of the centers of the three blobs.
- make_blobs:<br>
>The make_blobs function is a function that generates a set of points in a two-dimensional space.
## Synthesis Blocks
### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Signal 1 : sinusoidal signal
```python
s1 = np.sin(2 * time)
```

### notebooks/dataset2/calibration/plot_calibration.ipynb
CONTEXT:  Generate synthetic dataset   COMMENT:
```python
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)
y[: n_samples // 2] = 0
y[n_samples // 2 :] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])
```

## Code Concatenation
```python
s1 = np.sin(2 * time)
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)
y[: n_samples // 2] = 0
y[n_samples // 2 :] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])
```
