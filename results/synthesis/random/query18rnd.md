# Random Code Synthesis
Query `Compare structured vs unstructured Ward clustering.`
## Script Variables
- y_train:<br>
>It is a variable that contains the target variable of the dataset. It is a vector of integers that
- X_train:<br>
>X_train is a numpy array containing the training data. It is used to train the model and make
- all_models:<br>
>It is a dictionary that contains the Gradient Boosting Regressor models with different alpha values.
- coverage_fraction:<br>
>The coverage_fraction variable is used to calculate the coverage of the prediction of the model. It is calculated
- mean_pinball_loss:<br>
>The mean_pinball_loss function calculates the mean pinball loss of the predictions and the actual values.
- GradientBoostingRegressor:<br>
>GradientBoostingRegressor is a machine learning algorithm that uses gradient boosting to train a model. It is
- HalvingRandomSearchCV:<br>
>HalvingRandomSearchCV is a class that implements a hyperparameter optimization algorithm based on the Halving
- search_05p:<br>
>It is a variable that represents the probability that a randomly selected point from the training set belongs to the
- make_scorer:<br>
>It is a function that takes a scoring function as an argument and returns a scoring function that is a
- dict:<br>
>dict is a python dictionary which is a collection of key-value pairs. The key is a string or
- gbr:<br>
>It is a variable that contains the predictions of the model for the training data.
- alpha:<br>
>Alpha is the quantile of the loss function used in Gradient Boosting Regression. It is a value
- neg_mean_pinball_loss_05p_scorer:<br>
>The variable neg_mean_pinball_loss_05p_scorer is a function that calculates the negative
- metrics:<br>
>The variable metrics are used to evaluate the performance of the model in predicting the target variable. The metrics
- param_grid:<br>
>Param_grid is a dictionary that contains the parameters that will be searched for the Gradient Boosting Regress
- pprint:<br>
>pprint() is a function that is used to pretty print the output of a variable. It is
- y:<br>
>The variable y is the predicted probability of the data point being a 1 (positive) or
- X_transformed:<br>
>X_transformed is a matrix of 1000 rows and 100 columns. It contains the original
- nb:<br>
>The variable nb is a BernoulliNB() object. It is a type of classification algorithm that
- BernoulliNB:<br>
>BernoulliNB is a class in scikit-learn which is used to fit a Bernou
- plot:<br>
>Plot is a variable that is used to plot the data points in the given dataset. It is used
- X:<br>
>X is a 2x1500 matrix. It is the concatenation of two columns, x
- plt:<br>
>plt is a python library that is used for plotting data. It is used to create graphs and charts
- DBSCAN:<br>
>DBSCAN is a clustering algorithm that uses the distance between points to determine whether they belong to the same
- scale:<br>
>The variable scale is used to control the sensitivity of the HDBSCAN algorithm to the noise in the
- dbs:<br>
>The dbs variable is a DBSCAN object. It is used to perform clustering on the given dataset.
- fig:<br>
>The variable fig is a figure object that is created using the plt.subplots() function. The function takes
- np:<br>
>np is a library in python that provides a large collection of mathematical functions and data structures. It is
- x:<br>
>The variable x is a list of integers that represents the RGB values of the colors in the image.
- cv_score:<br>
>cv_score is the score of the model on the validation data. It is used to determine the best
- cv_best_iter:<br>
>cv_best_iter is the number of iterations that the model has been trained on the cross-validation set.
- first_week:<br>
>It is a variable that is used to select the first week of the data. It is used to
- quantiles:<br>
>The variable quantiles is used to specify the quantiles for the quantile loss function. The quant
- ax:<br>
>The variable ax is a matplotlib axis object that is used to plot the predicted and recorded average energy transfer
- len:<br>
>The len() function returns the length of an object. It is used to find the number of items
- _:<br>
>The variable _ is a placeholder for the number of iterations of the gradient boosting algorithm. It is used
- range:<br>
>The max_iter_list variable is a list of integers that represent the maximum number of iterations for the Hist
- HistGradientBoostingRegressor:<br>
>HistGradientBoostingRegressor is a machine learning algorithm that uses historical data to predict future values. It
- X_test:<br>
>X_test is a dataset containing the features of the test samples. It is used to predict the probability
- i:<br>
>i is a variable that is used to represent the missing values in the dataset. It is used to
- score:<br>
>The variable score is a measure of the performance of the model in predicting the energy transfer. It is
- predictions:<br>
>The variable predictions are used to predict the daily energy transfer based on the training data. The predictions are
- y_test:<br>
>y_test is a numpy array containing the actual values of the energy transfer for each day of the week
- common_params:<br>
>It is a dictionary that contains the parameters that are common to all the trees in the ensemble. These
- y_pred:<br>
>It is a variable that stores the predicted values of the energy transfer for each day of the week.
- hgbt_quantile:<br>
>hgbt_quantile is a variable that represents the quantile loss of the HistGradientBoosting
- colors:<br>
>The variable colors is a list of colors that are used to represent the different quantiles of the data
- n_clusters:<br>
>n_clusters is a variable that is used to specify the number of clusters to be formed in the clustering
- index:<br>
>The variable index is a variable that is used to store the index of the data points in the dataset
- AgglomerativeClustering:<br>
>AgglomerativeClustering is a clustering algorithm that uses a bottom-up approach to form clusters. It
- linkage:<br>
>Variable linkage is a method used to identify the relationship between two or more variables. It is used to
- kneighbors_graph:<br>
>kneighbors_graph is a function that creates a graph of the k nearest neighbors of each point in the
- t0:<br>
>t0 is a variable that represents the time taken for the execution of the script.
- elapsed_time:<br>
>Elapsed time is the time taken by the program to execute a block of code. It is measured in
- model:<br>
>The variable model is a model that is used to predict the output of a function based on its input
- connectivity:<br>
>Connectivity is a variable that is used to determine the distance between two points in a graph. It
- time:<br>
>The variable time is used to measure the execution time of the script. It is used to calculate the
- enumerate:<br>
>Enumerate is a function that returns a sequence of tuples, where each tuple contains the index of the
- sample_weight:<br>
>Sample weight is a variable that is used to indicate the importance of each sample in the dataset. It
- sw_train:<br>
>sw_train is a variable that is used to calculate the weights of the training data. It is used
- clf:<br>
>clf is a classifier that is used to predict the probability of a positive class. It is a calibrated
- prob_pos_isotonic:<br>
>Prob_pos_isotonic is a variable that contains the probability of a positive class for each sample
- CalibratedClassifierCV:<br>
>CalibratedClassifierCV is a class that calibrates a classifier using isotonic regression. It is
- clf_isotonic:<br>
>It is a classifier that uses the isotonic method to calibrate the predictions of the classifier clf.
## Synthesis Blocks
### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: On the test set, the estimated confidence interval is slightly too narrow. Note, however, that we would need to wrap those metrics in a
cross-validation loop to assess their variability under data resampling.   Tuning the hyper-parameters of the quantile regressors  In the plot above,
we observed that the 5th percentile regressor seems to underfit and could not adapt to sinusoidal shape of the signal.  The hyper-parameters of the
model were approximately hand-tuned for the median regressor and there is no reason that the same hyper-parameters are suitable for the 5th percentile
regressor.  To confirm this hypothesis, we tune the hyper-parameters of a new regressor of the 5th percentile by selecting the best model parameters
by cross-validation on the pinball loss with alpha=0.05:   COMMENT:
```python
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv

from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingRandomSearchCV
param_grid = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
)
alpha = 0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,

)
gbr = GradientBoostingRegressor(loss="quantiles", alpha=alpha, random_state=0)
search_05p = HalvingRandomSearchCV(
    gbr,
    param_grid,
    resource="n_estimators",
    max_resources=250,
    min_resources=50,
    scoring=neg_mean_pinball_loss_05p_scorer,
    n_jobs=2,
    random_state=0,
).fit(X_train, y_train)
pprint(search_05p.best_params_)
```

### notebooks/dataset2/ensemble_methods/plot_hgbt_regression.ipynb
CONTEXT: As expected, the model degrades as the proportion of missing values increases.   Support for quantile loss  The quantile loss in regression
enables a view of the variability or uncertainty of the target variable. For instance, predicting the 5th and 95th percentiles can provide a 90%
prediction interval, i.e. the range within which we expect a new observed value to fall with 90% probability.   COMMENT:
```python
from sklearn.metrics import mean_pinball_loss
quantiles = [0.95, 0.05]
predictions = []
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.values[first_week], label="Actual transfer")
for quantiles in quantiles:
    hgbt_quantile = HistGradientBoostingRegressor(
        loss="quantiles", quantiles=quantiles, **common_params
    )
    hgbt_quantile.fit(X_train, y_train)
    y_pred = hgbt_quantile.predict(X_test[first_week])
    predictions.append(y_pred)
    score = mean_pinball_loss(y_test[first_week], y_pred)
    ax.plot(
        y_pred[first_week],
        label=f"quantiles={quantiles}, pinball loss={score:.2f}",
        alpha=0.5,
    )
ax.fill_between(
    range(len(predictions[0][first_week])),
    predictions[0][first_week],
    predictions[1][first_week],
    color=colors[0],
    alpha=0.1,
)
ax.set(
    title="Daily energy transfer predictions with quantiles loss",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
_ = ax.legend(loc="lower right")
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_oob.ipynb
CONTEXT:   Gradient Boosting Out-of-Bag estimates Out-of-bag (OOB) estimates can be a useful heuristic to estimate the "optimal" number of boosting
iterations. OOB estimates are almost identical to cross-validation estimates but they can be computed on-the-fly without the need for repeated model
fitting. OOB estimates are only available for Stochastic Gradient Boosting (i.e. ``subsample < 1.0``), the estimates are derived from the improvement
in loss based on the examples not included in the bootstrap sample (the so-called out-of-bag examples). The OOB estimator is a pessimistic estimator
of the true test loss, but remains a fairly good approximation for a small number of trees. The figure shows the cumulative sum of the negative OOB
improvements as a function of the boosting iteration. As you can see, it tracks the test loss for the first hundred iterations but then diverges in a
pessimistic way. The figure also shows the performance of 3-fold cross validation which usually gives a better estimate of the test loss but is
computationally more demanding.  COMMENT: min loss according to cv (normalize such that first loss is 0)
```python
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]
```

### notebooks/dataset2/ensemble_methods/plot_random_forest_embedding.ipynb
CONTEXT:   Hashing feature transformation using Totally Random Trees  RandomTreesEmbedding provides a way to map data to a very high-dimensional,
sparse representation, which might be beneficial for classification. The mapping is completely unsupervised and very efficient.  This example
visualizes the partitions given by several trees and shows how the transformation can also be used for non-linear dimensionality reduction or non-
linear classification.  Points that are neighboring often share the same leaf of a tree and therefore share large parts of their hashed
representation. This allows to separate two concentric circles simply based on the principal components of the transformed data with truncated SVD.
In high-dimensional spaces, linear classifiers often achieve excellent accuracy. For sparse binary data, BernoulliNB is particularly well-suited. The
bottom row compares the decision boundary obtained by BernoulliNB in the transformed space with an ExtraTreesClassifier forests learned on the
original data.  COMMENT: Learn a Naive Bayes classifier on the transformed data
```python
nb = BernoulliNB()
nb.fit(X_transformed, y)
```

### notebooks/dataset2/ensemble_methods/plot_gradient_boosting_quantile.ipynb
CONTEXT: Errors are higher meaning the models slightly overfitted the data. It still shows that the best test metric is obtained when the model is
trained by minimizing this same metric.  Note that the conditional median estimator is competitive with the squared error estimator in terms of MSE on
the test set: this can be explained by the fact the squared error estimator is very sensitive to large outliers which can cause significant
overfitting. This can be seen on the right hand side of the previous plot. The conditional median estimator is biased (underestimation for this
asymmetric noise) but is also naturally robust to outliers and overfits less.    Calibration of the confidence interval  We can also evaluate the
ability of the two extreme quantile estimators at producing a well-calibrated conditional 90%-confidence interval.  To do this we can compute the
fraction of observations that fall between the predictions:   COMMENT:
```python
coverage_fraction(
    y_train,
    all_models["q 0.05"].predict(X_train),
    all_models["q 0.95"].predict(X_train),
)
```

### notebooks/dataset2/clustering/plot_hdbscan.ipynb
CONTEXT: Indeed, in order to maintain the same results we would have to scale `eps` by the same factor.   COMMENT:
```python
fig, plot = plt.subplots(1, 1, figsize=(12, 5))
dbs = DBSCAN(eps=0.9).fit(3 * X)
plot(3 * X, dbs.labels_, parameters={"scale": 3, "eps": 0.9}, ax=plot)
```

### notebooks/dataset2/clustering/plot_hdbscan.ipynb
CONTEXT: Indeed, in order to maintain the same results we would have to scale `eps` by the same factor.   COMMENT:
```python
fig, plot = plt.subplots(1, 1, figsize=(12, 5))
dbs = DBSCAN(eps=0.9).fit(3 * X)
plot(3 * X, dbs.labels_, parameters={"scale": 3, "eps": 0.9}, ax=plot)
```

### notebooks/dataset2/calibration/plot_calibration.ipynb
CONTEXT:  Gaussian Naive-Bayes   COMMENT: With no calibration
```python
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
```

### notebooks/dataset2/clustering/plot_agglomerative_clustering.ipynb
CONTEXT:   Agglomerative clustering with and without structure  This example shows the effect of imposing a connectivity graph to capture local
structure in the data. The graph is simply the graph of 20 nearest neighbors.  There are two advantages of imposing a connectivity. First, clustering
with sparse connectivity matrices is faster in general.  Second, when using a connectivity matrix, single, average and complete linkage are unstable
and tend to create a few clusters that grow very quickly. Indeed, average and complete linkage fight this percolation behavior by considering all the
distances between two clusters when merging them ( while single linkage exaggerates the behaviour by considering only the shortest distance between
clusters). The connectivity graph breaks this mechanism for average and complete linkage, making them resemble the more brittle single linkage. This
effect is more pronounced for very sparse graphs (try decreasing the number of neighbors in kneighbors_graph) and with complete linkage. In
particular, having a very small number of neighbors in the graph, imposes a geometry that is close to that of single linkage, which is well known to
have this percolation instability.  COMMENT: Create a graph capturing local connectivity. Larger number of neighbors will give more homogeneous
clusters to the cost of computation time. A very large number of neighbors gives more evenly distributed cluster sizes, but may not impose the local
manifold structure of the data
```python
kneighbors_graph = kneighbors_graph(X, 30, include_self=False)
for connectivity in (None, kneighbors_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(("average", "complete", "ward", "single")):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(
                linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
            )
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
            plt.title(
                "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
                fontdict=dict(verticalalignment="top"),
            )
            plt.plot("equal")
            plt.plot("off")
            plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
            plt.suptitle(
                "n_cluster=%i, connectivity=%r"
                % (n_clusters, connectivity is not None),
                size=17,
            )
plt.show()
```

## Code Concatenation
```python
from pprint import pprint
from sklearn.experimental import enable_halving_search_cv

from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingRandomSearchCV
param_grid = dict(
    learning_rate=[0.05, 0.1, 0.2],
    max_depth=[2, 5, 10],
    min_samples_leaf=[1, 5, 10, 20],
    min_samples_split=[5, 10, 20, 30, 50],
)
alpha = 0.05
neg_mean_pinball_loss_05p_scorer = make_scorer(
    mean_pinball_loss,
    alpha=alpha,
    greater_is_better=False,

)
gbr = GradientBoostingRegressor(loss="quantiles", alpha=alpha, random_state=0)
search_05p = HalvingRandomSearchCV(
    gbr,
    param_grid,
    resource="n_estimators",
    max_resources=250,
    min_resources=50,
    scoring=neg_mean_pinball_loss_05p_scorer,
    n_jobs=2,
    random_state=0,
).fit(X_train, y_train)
pprint(search_05p.best_params_)
from sklearn.metrics import mean_pinball_loss
quantiles = [0.95, 0.05]
predictions = []
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test.values[first_week], label="Actual transfer")
for quantiles in quantiles:
    hgbt_quantile = HistGradientBoostingRegressor(
        loss="quantiles", quantiles=quantiles, **common_params
    )
    hgbt_quantile.fit(X_train, y_train)
    y_pred = hgbt_quantile.predict(X_test[first_week])
    predictions.append(y_pred)
    score = mean_pinball_loss(y_test[first_week], y_pred)
    ax.plot(
        y_pred[first_week],
        label=f"quantiles={quantiles}, pinball loss={score:.2f}",
        alpha=0.5,
    )
ax.fill_between(
    range(len(predictions[0][first_week])),
    predictions[0][first_week],
    predictions[1][first_week],
    color=colors[0],
    alpha=0.1,
)
ax.set(
    title="Daily energy transfer predictions with quantiles loss",
    xticks=[(i + 0.2) * 48 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Time of the week",
    ylabel="Normalized energy transfer",
)
_ = ax.legend(loc="lower right")
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]
nb = BernoulliNB()
nb.fit(X_transformed, y)
coverage_fraction(
    y_train,
    all_models["q 0.05"].predict(X_train),
    all_models["q 0.95"].predict(X_train),
)
fig, plot = plt.subplots(1, 1, figsize=(12, 5))
dbs = DBSCAN(eps=0.9).fit(3 * X)
plot(3 * X, dbs.labels_, parameters={"scale": 3, "eps": 0.9}, ax=plot)
fig, plot = plt.subplots(1, 1, figsize=(12, 5))
dbs = DBSCAN(eps=0.9).fit(3 * X)
plot(3 * X, dbs.labels_, parameters={"scale": 3, "eps": 0.9}, ax=plot)
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
kneighbors_graph = kneighbors_graph(X, 30, include_self=False)
for connectivity in (None, kneighbors_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(("average", "complete", "ward", "single")):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(
                linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
            )
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
            plt.title(
                "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
                fontdict=dict(verticalalignment="top"),
            )
            plt.plot("equal")
            plt.plot("off")
            plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
            plt.suptitle(
                "n_cluster=%i, connectivity=%r"
                % (n_clusters, connectivity is not None),
                size=17,
            )
plt.show()
```
