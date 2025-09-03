# Reverse Embedding Code Synthesis
Query `Analyze forest embedding of iris dataset.`
## Script Variables
- iris:<br>
>It is a dataset that contains information about 150 flowers of three different species of iris. The dataset
- X:<br>
>X is a numpy array of shape (150, 2) containing the first two features of the
## Synthesis Blocks
### notebooks/dataset2/classification/plot_classification_probability.ipynb
CONTEXT:  Data: 2D projection of the iris dataset   COMMENT: we only take the first two features for visualization
```python
X = iris.data[:, 0:2]
```

## Code Concatenation
```python
X = iris.data[:, 0:2]
```
