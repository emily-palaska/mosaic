# String Code Synthesis
Query `How do you normalize data?`
## Script Variables
- x_train:<br>
>x_train is a numpy array of size (n_samples, n_features) containing the training data. The values of x_train are the features of the training data. The values of y_train are the labels of the training data.
- preprocessing:<br>
>The preprocessing variable is used to normalize the data. It is used to remove the outliers and to make the data more robust. The data is normalized by subtracting the mean and dividing by the standard deviation. This helps to reduce the impact of outliers and to make the data more robust. The normalization process is done before the data is used for training the model.
## Synthesis Blocks
### notebooks/example_more.ipynb
CONTEXT: def train_lr(x_train, y_train, preprocessing="normalize"): COMMENT: Standardize training data
```python
if preprocessing == "standardize":
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
```

### notebooks/example_more.ipynb
CONTEXT: def train_lr(x_train, y_train, preprocessing="normalize"): COMMENT: Normalize training data
```python
if preprocessing == "normalize":
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
```

## Code Concatenation
```python
if preprocessing == "standardize":
    x_train = (x_train - x_train.mean(axis=0)) / (x_train.std(axis=0))
if preprocessing == "normalize":
    x_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) - x_train.min(axis=0))
```
