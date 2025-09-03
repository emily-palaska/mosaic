# Reverse Embedding Code Synthesis
Query `Use post-pruning on decision tree.`
## Script Variables
- DecisionTreeClassifier:<br>
>DecisionTreeClassifier is a machine learning algorithm that is used for classification problems. It is a tree-based
- load_breast_cancer:<br>
>The load_breast_cancer function is used to load the breast cancer dataset from the scikit
- train_test_split:<br>
>It is a function that is used to split the dataset into training and testing sets. It takes in
- plt:<br>
>plt is a python library that is used for plotting data in a variety of ways. It is a
## Synthesis Blocks
### notebooks/dataset2/decision_trees/plot_cost_complexity_pruning.ipynb
CONTEXT:   Post pruning decision trees with cost complexity pruning  .. currentmodule:: sklearn.tree  The :class:`DecisionTreeClassifier` provides
parameters such as ``min_samples_leaf`` and ``max_depth`` to prevent a tree from overfiting. Cost complexity pruning provides another option to
control the size of a tree. In :class:`DecisionTreeClassifier`, this pruning technique is parameterized by the cost complexity parameter,
``ccp_alpha``. Greater values of ``ccp_alpha`` increase the number of nodes pruned. Here we only show the effect of ``ccp_alpha`` on regularizing the
trees and how to choose a ``ccp_alpha`` based on validation scores.  See also `minimal_cost_complexity_pruning` for details on pruning.  COMMENT:
Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

## Code Concatenation
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```
