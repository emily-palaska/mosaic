# Embedding Code Synthesis
Query `Run GMM on sinusoidal synthetic data.`
## Script Variables
- time:<br>
>The variable time is a numpy array that contains 2000 values ranging from 0 to 8
- np:<br>
>The np is a Python package that provides a high-performance multidimensional array object, and tools for working
- s1:<br>
>The variable s1 is a sine function of time. It is used to generate a sine wave that
## Synthesis Blocks
### notebooks/dataset2/decomposition/plot_ica_blind_source_separation.ipynb
CONTEXT:  Generate sample data   COMMENT: Signal 1 : sinusoidal signal
```python
s1 = np.sin(2 * time)
```

## Code Concatenation
```python
s1 = np.sin(2 * time)
```
