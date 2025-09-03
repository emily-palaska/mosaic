# Embedding Code Synthesis
Query `Build feature selection pipeline.`
## Script Variables
- anova_svm:<br>
>AnovaSVM is a class that implements the anova-based SVM algorithm.
- X_train:<br>
>X_train is a 2D numpy array of shape (n_samples, n_features) where n
- clf:<br>
>clf is a LinearSVC classifier which is a linear support vector machine classifier. It is used to
- make_pipeline:<br>
>The make_pipeline function is used to create a pipeline of estimators. It takes a list of estim
- LinearSVC:<br>
>LinearSVC is a linear classifier that uses the L1 norm for regularization.
- y_train:<br>
>It is a vector of labels that indicate the class of each sample in the training set.
- anova_filter:<br>
>It is a feature selection method that uses the F-statistic to rank features based on their statistical significance
- f_classif:<br>
>f_classif is a function that calculates the F-statistic for the given data. It is used
- SelectKBest:<br>
>SelectKBest is a feature selection method that selects the best k features based on a given scoring function
- RandomForestClassifier:<br>
>RandomForestClassifier is a machine learning algorithm that uses a decision tree to classify data. It is a
- AdaBoostClassifier:<br>
>AdaBoostClassifier is a machine learning algorithm that uses a combination of weak classifiers to create a strong classifier
- y:<br>
>The variable y is a 2D array of size (25000, 1) which contains
- plt:<br>
>plt is a module in python which is used to create plots in python. It is a library for
- train_test_split:<br>
>It is a function that splits the dataset into training and testing sets. The test_size parameter specifies the
- datasets:<br>
>The datasets are used to train the model and predict the class of the data points.
- RBF:<br>
>RBF stands for Radial Basis Function. It is a kernel function used in Gaussian Process Classifier.
- make_moons:<br>
>The make_moons function is used to generate a dataset of 2D points that are linearly
- rng:<br>
>The variable rng is a random number generator. It is used to generate random numbers for the script.
- i:<br>
>i is a variable that is used to represent the current dataset being plotted. It is used to create
- names:<br>
>names
- StandardScaler:<br>
>StandardScaler is a class that is used to standardize the features of a dataset. It is used
- KNeighborsClassifier:<br>
>KNeighborsClassifier is a classifier that uses a k-NN algorithm for classification. It is a simple
- classifiers:<br>
>The variable classifiers are used to classify the data into different categories. They are used to identify the patterns
- DecisionBoundaryDisplay:<br>
>The DecisionBoundaryDisplay class is used to visualize the decision boundary of a classifier. It takes in a
- ListedColormap:<br>
>ListedColormap is a class that provides a list of colors for the colormap.
- GaussianNB:<br>
>GaussianNB is a classifier that uses Bayes' theorem with Gaussian distributions. It is a simple
- X:<br>
>X is a numpy array of shape (n_samples, n_features) containing the input samples. It
- np:<br>
>Numpy is a Python library that provides a multidimensional array object, along with a collection of mathematical
- GaussianProcessClassifier:<br>
>GaussianProcessClassifier is a machine learning algorithm that uses a Gaussian process to model the relationship between the
- make_circles:<br>
>The make_circles function is used to generate a dataset of circles with varying radii and centers.
- DecisionTreeClassifier:<br>
>DecisionTreeClassifier is a classification algorithm that uses a decision tree to classify data. It is a simple
- MLPClassifier:<br>
>MLPClassifier is a classifier that uses a multi-layer perceptron (MLP) to learn a
- linearly_separable:<br>
>It is a tuple containing two numpy arrays, X and y, where X is a matrix of n
- make_classification:<br>
>The make_classification function is a function that generates a dataset for classification tasks. It takes in a number
- figure:<br>
>The variable figure is a figure object that is used to display the results of the script. It is
- SVC:<br>
>SVC stands for Support Vector Classifier. It is a supervised machine learning algorithm that is used for classification
- QuadraticDiscriminantAnalysis:<br>
>QuadraticDiscriminantAnalysis is a classifier that uses quadratic discriminant analysis to classify data. It
- make_blobs:<br>
>It is a function that generates a dataset of 25000 samples with 2 clusters of 5
- n_centers:<br>
>n_centers is a 2D array of shape (100, 2) that contains the x
- t_batch:<br>
>t_batch is the time taken by the k-means algorithm to complete its execution. It is a
- k:<br>
>k is a variable that is used to represent the number of clusters that are present in the data set
- k_means:<br>
>k_means is a variable that is used to perform K-means clustering on the dataset X. It
- time:<br>
>The variable time is used to measure the time taken by the MiniBatchKMeans algorithm to complete its
- t0:<br>
>t0 is a variable that stores the time taken by the script to run the KMeans algorithm.
- KMeans:<br>
>KMeans is a clustering algorithm that is used to group similar data points together. It is a type
## Synthesis Blocks
### notebooks/dataset2/feature_selection/plot_feature_selection_pipeline.ipynb
CONTEXT: We will start by generating a binary classification dataset. Subsequently, we will divide the dataset into two subsets.   COMMENT:
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
anova_filter = SelectKBest(f_classif, k=3)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)
```

### notebooks/dataset2/classification/plot_classifier_comparison.ipynb
CONTEXT:   Classifier comparison  A comparison of several classifiers in scikit-learn on synthetic datasets. The point of this example is to
illustrate the nature of decision boundaries of different classifiers. This should be taken with a grain of salt, as the intuition conveyed by these
examples does not necessarily carry over to real datasets.  Particularly in high-dimensional spaces, data can more easily be separated linearly and
the simplicity of classifiers such as naive Bayes and linear SVMs might lead to better generalization than is achieved by other classifiers.  The
plots show training points in solid colors and testing points semi-transparent. The lower right shows the classification accuracy on the test set.
COMMENT: Authors: The scikit-learn developers SPDX-License-Identifier: BSD-3-Clause
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]
figure = plt.figure(figsize=(27, 9))
i = 1
```

### notebooks/dataset2/clustering/plot_mini_batch_kmeans.ipynb
CONTEXT:  Compute clustering with KMeans   COMMENT:
```python
import time
from sklearn.cluster import KMeans
k_means = KMeans(init="k-means++", k=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
```

### notebooks/dataset2/clustering/plot_birch_vs_minibatchkmeans.ipynb
CONTEXT:   Compare BIRCH and MiniBatchKMeans  This example compares the timing of BIRCH (with and without the global clustering step) and
MiniBatchKMeans on a synthetic dataset having 25,000 samples and 2 features generated using make_blobs.  Both ``MiniBatchKMeans`` and ``BIRCH`` are
very scalable algorithms and could run efficiently on hundreds of thousands or even millions of datapoints. We chose to limit the dataset size of this
example in the interest of keeping our Continuous Integration resource usage reasonable but the interested reader might enjoy editing this script to
rerun it with a larger value for `n_samples`.  If ``n_clusters`` is set to None, the data is reduced from 25,000 samples to a set of 158 clusters.
This can be viewed as a preprocessing step before the final (global) clustering step that further reduces these 158 clusters to 100 clusters.
COMMENT: Generate blobs to do a comparison between MiniBatchKMeans and BIRCH.
```python
X, y = make_blobs(n_samples=25000, centers=n_centers, random_state=0)
```

## Code Concatenation
```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
anova_filter = SelectKBest(f_classif, k=3)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
datasets = [
    make_moons(noise=0.3, random_state=0),
    make_circles(noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]
figure = plt.figure(figsize=(27, 9))
i = 1
import time
from sklearn.cluster import KMeans
k_means = KMeans(init="k-means++", k=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0
X, y = make_blobs(n_samples=25000, centers=n_centers, random_state=0)
```
