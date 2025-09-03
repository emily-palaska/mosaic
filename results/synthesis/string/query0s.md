# String Code Synthesis
Query `Display internal structure of a tree model.`
## Script Variables
- n_classes:<br>
>It is a constant that defines the number of classes in the dataset. In this case, it is
- plot_colors:<br>
>It is a list of colors used to represent the classes in the plot. The colors are defined by
- plot_step:<br>
>It is a value that is used to determine the step size of the plot. This value is used
- clf:<br>
>clf is a decision tree classifier. It is a supervised learning algorithm that can be used to predict the
- node_indicator:<br>
>It is a numpy array that contains the number of common nodes between each pair of samples.
- sample_id:<br>
>It is a variable that stores the id of the sample. It is used to identify the sample in
- X_test:<br>
>X_test is a 2D array that contains the test data. It has two dimensions, one
- leaf_id:<br>
>It is a variable that stores the leaf id of each sample in the test set. The leaf id
- node_depth:<br>
>Node depth is a variable that is used to keep track of the depth of each node in the tree
- stack:<br>
>The stack is a list of tuples that stores the node number and the depth of the node. The
- node_id:<br>
>The variable node_id is used to identify the node in the tree that the current sample belongs to.
- len:<br>
>len is a built-in function in Python that returns the length of an iterable object. In this case
## Synthesis Blocks
### notebooks/dataset2/decision_trees/plot_iris_dtc.ipynb
CONTEXT: Display the decision functions of trees trained on all pairs of features.   COMMENT:
```python
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
```

### notebooks/dataset2/decision_trees/plot_unveil_tree_structure.ipynb
CONTEXT:  Decision path  We can also retrieve the decision path of samples of interest. The ``decision_path`` method outputs an indicator matrix that
allows us to retrieve the nodes the samples of interest traverse through. A non zero element in the indicator matrix at position ``(i, j)`` indicates
that the sample ``i`` goes through the node ``j``. Or, for one sample ``i``, the positions of the non zero elements in row ``i`` of the indicator
matrix designate the ids of the nodes that sample goes through.  The leaf ids reached by samples of interest can be obtained with the ``apply``
method. This returns an array of the node ids of the leaves reached by each sample of interest. Using the leaf ids and the ``decision_path`` we can
obtain the splitting conditions that were used to predict a sample or a group of samples. First, let's do it for one sample. Note that ``node_index``
is a sparse matrix.   COMMENT:
```python
node_indicator = clf.decision_path(X_test)
leaf_id = clf.apply(X_test)
sample_id = 0
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
compute the depth of each node and whether or not it is a leaf.   COMMENT: `pop` ensures each node is only visited once
```python
while len(stack) > 0:
    node_id, node_depth = stack.pop()
    node_depth[node_id] = node_depth
```

## Code Concatenation
```python
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
node_indicator = clf.decision_path(X_test)
leaf_id = clf.apply(X_test)
sample_id = 0
while len(stack) > 0:
    node_id, node_depth = stack.pop()
    node_depth[node_id] = node_depth
```
