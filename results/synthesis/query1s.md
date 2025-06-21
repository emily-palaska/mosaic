# String Code Synthesis
Query `Create a regression model.`
## Script Variables
- model:<br>
>The model is a machine learning model that is used to train a model on the given data. The model takes in the input data and the target data and uses this information to learn a function that maps the input data to the target data. The model is then used to make predictions on new data.
- LogisticRegression:<br>
>LogisticRegression is a classification algorithm that is used to predict the probability of a given data point belonging to a particular class. It is based on the logistic function, which is a sigmoid function that maps the input data to a probability value between 0 and 1. The output of the logistic function is then used to predict the class of the data point.
## Synthesis Blocks
### notebooks/example_more.ipynb
CONTEXT: def train_lr(x_train, y_train, preprocessing="normalize"): COMMENT: create a logistic regression model
```python
model = LogisticRegression()
```

## Code Concatenation
```python
model = LogisticRegression()
```
