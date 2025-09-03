# Embedding Code Synthesis
Query `Display internal structure of a tree model.`
## Script Variables
- n_classes:<br>
>It is a constant that defines the number of classes in the dataset. In this case, it is
- plot_colors:<br>
>It is a list of colors used to represent the classes in the plot. The colors are defined by
- plot_step:<br>
>It is a value that is used to determine the step size of the plot. This value is used
- anova_svm:<br>
>AnovaSVM is a class that implements the anova-based SVM algorithm.
- y_train_linear:<br>
>It is a numpy array containing the target values of the training data.
- make_pipeline:<br>
>The make_pipeline function is used to create a pipeline of steps. The steps are applied in the order
- random_tree_embedding:<br>
>The random_tree_embedding is a random forest classifier that uses a random subset of features to make predictions.
- pipeline:<br>
>The pipeline variable is used to store the pipeline object that is used to fit the model. The pipeline
- X_train_linear:<br>
>X_train_linear is a matrix of size (n_samples, n_features) where n_samples is the
- LogisticRegression:<br>
>LogisticRegression is a machine learning algorithm used for classification problems. It is a type of supervised learning
- rt_model:<br>
>The rt_model is a Random Forest model that is trained on the training data and used to predict the
- FunctionTransformer:<br>
>FunctionTransformer is a class in scikit-learn that allows us to transform data before it is used
- OneHotEncoder:<br>
>OneHotEncoder is a class that converts categorical features into binary features. It is used to encode categorical
- stack:<br>
>The stack is a list of tuples that stores the node number and the depth of the node. The
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_feature_transformation.ipynb
CONTEXT: Now, we will create three pipelines that will use the above embedding as a preprocessing stage.  The random trees embedding can be directly
pipelined with the logistic regression because it is a standard scikit-learn transformer.   COMMENT:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
rt_model.fit(X_train_linear, y_train_linear)
```

### notebooks/dataset2/ensemble_methods/plot_feature_transformation.ipynb
CONTEXT: Then, we can pipeline random forest or gradient boosting with a logistic regression. However, the feature transformation will happen by
calling the method `apply`. The pipeline in scikit-learn expects a call to `transform`. Therefore, we wrapped the call to `apply` within a
`FunctionTransformer`.   COMMENT:
```python
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
```

### notebooks/dataset2/decision_trees/plot_iris_dtc.ipynb
CONTEXT: Display the decision functions of trees trained on all pairs of features.   COMMENT:
```python
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
```

### notebooks/dataset2/decision_trees/plot_unveil_tree_structure.ipynb
CONTEXT:  Tree structure  The decision classifier has an attribute called ``tree_`` which allows access to low level attributes such as
``node_count``, the total number of nodes, and ``max_depth``, the maximal depth of the tree. The ``tree_.compute_node_depths()`` method computes the
depth of each node in the tree. `tree_` also stores the entire binary tree structure, represented as a number of parallel arrays. The i-th element of
each array holds information about the node ``i``. Node 0 is the tree's root. Some of the arrays only apply to either leaves or split nodes. In this
case the values of the nodes of the other type is arbitrary. For example, the arrays ``feature`` and ``threshold`` only apply to split nodes. The
values for leaf nodes in these arrays are therefore arbitrary.  Among these arrays, we have:  - ``children_left[i]``: id of the left child of node
``i`` or -1 if leaf node - ``children_right[i]``: id of the right child of node ``i`` or -1 if leaf node - ``feature[i]``: feature used for splitting
node ``i`` - ``threshold[i]``: threshold value at node ``i`` - ``n_node_samples[i]``: the number of training samples reaching node ``i`` -
``impurity[i]``: the impurity at node ``i`` - ``weighted_n_node_samples[i]``: the weighted number of training samples   reaching node ``i`` -
``value[i, j, k]``: the summary of the training samples that reached node i for   output j and class k (for regression tree, class is set to 1). See
below   for more information about ``value``.  Using the arrays, we can traverse the tree structure to compute various properties. Below, we will
compute the depth of each node and whether or not it is a leaf.   COMMENT: start with the root node id (0) and its depth (0)
```python
stack = [(0, 0)]
```

### notebooks/dataset2/feature_selection/plot_feature_selection_pipeline.ipynb
CONTEXT: Once the training is complete, we can predict on new unseen samples. In this case, the feature selector will only select the most
discriminative features based on the information stored during training. Then, the data will be passed to the classifier which will make the
prediction.  Here, we show the final metrics via a classification report.   COMMENT:
```python
anova_svm[-1].coef_
```

## Code Concatenation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
rt_model.fit(X_train_linear, y_train_linear)
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
stack = [(0, 0)]
anova_svm[-1].coef_
```
