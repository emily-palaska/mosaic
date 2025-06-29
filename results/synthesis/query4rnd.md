# Random Code Synthesis
Query `Simple PCA algorithm.`
## Script Variables
- pca:<br>
>pca is a PCA object that is used to reduce the dimensionality of the data. It does
- pcr:<br>
>The variable pcr is a pipeline that contains a standard scaler, a PCA component, and a linear
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

## Code Concatenation
```python
pca = pcr.named_steps["pca"]
```
