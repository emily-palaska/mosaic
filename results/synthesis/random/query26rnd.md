# Random Code Synthesis
Query `Split multilabel dataset into train and test.`
## Script Variables
- anova_svm:<br>
>AnovaSVM is a class that implements the anova-based SVM algorithm.
## Synthesis Blocks
### notebooks/dataset2/feature_selection/plot_feature_selection_pipeline.ipynb
CONTEXT: Be aware that you can inspect a step in the pipeline. For instance, we might be interested about the parameters of the classifier. Since we
selected three features, we expect to have three coefficients.   COMMENT:
```python
anova_svm[:-1].inverse_transform(anova_svm[-1].coef_)
```

## Code Concatenation
```python
anova_svm[:-1].inverse_transform(anova_svm[-1].coef_)
```
