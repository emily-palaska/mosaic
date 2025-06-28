# Random Code Synthesis
Query `PCA algorithm`
## Script Variables
- pca:<br>
>pca is a PCA object which is used to perform Principal Component Analysis. PCA is a dimensionality
- pcr:<br>
>The variable pcr is a function that calculates the r-squared value of a given dataset. It
## Synthesis Blocks
### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT: For the purpose of this example, we now define the target `y` such that it is strongly correlated with a direction that has a small variance.
To this end, we will project `X` onto the second component, and add some noise to it.   COMMENT:
```python
pca = pcr.named_steps["pca"]
```

## Code Concatenation
```python
pca = pcr.named_steps["pca"]
pca = pcr.named_steps["pca"]
```
