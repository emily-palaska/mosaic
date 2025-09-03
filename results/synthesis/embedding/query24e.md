# Embedding Code Synthesis
Query `Compare gradient boosting categorical method.`
## Script Variables
- hist_native:<br>
>The variable hist_native is a HistGradientBoostingRegressor object that is used to fit a gradient boosting
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses gradient boosting to fit a series of decision
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_categorical.ipynb
CONTEXT:  Gradient boosting estimator with native categorical support We now create a :class:`~ensemble.HistGradientBoostingRegressor` estimator that
will natively handle categorical features. This estimator will not treat categorical features as ordered quantities. We set
`categorical_features="from_dtype"` such that features with categorical dtype are considered categorical features.  The main difference between this
estimator and the previous one is that in this one, we let the :class:`~ensemble.HistGradientBoostingRegressor` detect which features are categorical
from the DataFrame columns' dtypes.   COMMENT:
```python
hist_native = HistGradientBoostingRegressor(
    random_state=42, categorical_features="from_dtype"
)
```

## Code Concatenation
```python
hist_native = HistGradientBoostingRegressor(
    random_state=42, categorical_features="from_dtype"
)
```
