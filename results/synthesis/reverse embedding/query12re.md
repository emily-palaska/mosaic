# Reverse Embedding Code Synthesis
Query `Compare PCR and PLS regression results.`
## Script Variables
- X_test:<br>
>X_test is a matrix of 2 columns and 100 rows. It contains the values of the
- y_test:<br>
>The variable y_test is a vector of test data that is used to evaluate the performance of the model
- pcr:<br>
>pcr is a Pipeline object that is used to perform a PCA analysis on the input data. It
- print:<br>
>The print function is used to display the output of the script to the console. It takes a single
- pls:<br>
>pls is a variable that is used to store the PLS regression model. It is used to predict
- cachedir:<br>
>The variable cachedir is a temporary directory that is used to store the memory cache of the BayesianR
- shutil:<br>
>shutil is a module in Python that provides a number of functions for working with files and directories.
- np:<br>
>np is a python library that provides a large set of mathematical functions and data structures. It is used
- matplotlib:<br>
>Matplotlib is a Python library that is used for data visualization. It provides a wide range of plotting
- MinCovDet:<br>
>The MinCovDet is a class that implements the Minimum Covariance Determinant (MCD)
- plt:<br>
>plt is a python library that is used to create plots. It is a part of the matplotlib library
- EmpiricalCovariance:<br>
>EmpiricalCovariance is a class that calculates the empirical covariance matrix of a given dataset. It
## Synthesis Blocks
### notebooks/dataset2/covariance_estimation/plot_robust_vs_empirical_covariance.ipynb
CONTEXT:   Robust vs Empirical covariance estimate  The usual covariance maximum likelihood estimate is very sensitive to the presence of outliers in
the data set. In such a case, it would be better to use a robust estimator of covariance to guarantee that the estimation is resistant to "erroneous"
observations in the data set. [1]_, [2]_   Minimum Covariance Determinant Estimator The Minimum Covariance Determinant estimator is a robust, high-
breakdown point (i.e. it can be used to estimate the covariance matrix of highly contaminated datasets, up to $\frac{n_\text{samples} -
n_\text{features}-1}{2}$ outliers) estimator of covariance. The idea is to find $\frac{n_\text{samples} + n_\text{features}+1}{2}$ observations whose
empirical covariance has the smallest determinant, yielding a "pure" subset of observations from which to compute standards estimates of location and
covariance. After a correction step aiming at compensating the fact that the estimates were learned from only a portion of the initial data, we end up
with robust estimates of the data set location and covariance.  The Minimum Covariance Determinant estimator (MCD) has been introduced by P.J.Rousseuw
in [3]_.   Evaluation In this example, we compare the estimation errors that are made when using various types of location and covariance estimates on
contaminated Gaussian distributed data sets:  - The mean and the empirical covariance of the full dataset, which break   down as soon as there are
outliers in the data set - The robust MCD, that has a low error provided   $n_\text{samples} > 5n_\text{features}$ - The mean and the empirical
covariance of the observations that are known   to be good ones. This can be considered as a "perfect" MCD estimation,   so one can trust our
implementation by comparing to this case.    References .. [1] Johanna Hardin, David M Rocke. The distribution of robust distances.     Journal of
Computational and Graphical Statistics. December 1, 2005,     14(4): 928-946. .. [2] Zoubir A., Koivunen V., Chakhchoukh Y. and Muma M. (2012). Robust
estimation in signal processing: A tutorial-style treatment of     fundamental concepts. IEEE Signal Processing Magazine 29(4), 61-80. .. [3] P. J.
Rousseeuw. Least median of squares regression. Journal of American     Statistical Ass., 79:871, 1984.  COMMENT: Authors: The scikit-learn developers
SPDX-License-Identifier: BSD-3-Clause
```python
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
```

### notebooks/dataset2/covariance_estimation/plot_robust_vs_empirical_covariance.ipynb
CONTEXT:   Robust vs Empirical covariance estimate  The usual covariance maximum likelihood estimate is very sensitive to the presence of outliers in
the data set. In such a case, it would be better to use a robust estimator of covariance to guarantee that the estimation is resistant to "erroneous"
observations in the data set. [1]_, [2]_   Minimum Covariance Determinant Estimator The Minimum Covariance Determinant estimator is a robust, high-
breakdown point (i.e. it can be used to estimate the covariance matrix of highly contaminated datasets, up to $\frac{n_\text{samples} -
n_\text{features}-1}{2}$ outliers) estimator of covariance. The idea is to find $\frac{n_\text{samples} + n_\text{features}+1}{2}$ observations whose
empirical covariance has the smallest determinant, yielding a "pure" subset of observations from which to compute standards estimates of location and
covariance. After a correction step aiming at compensating the fact that the estimates were learned from only a portion of the initial data, we end up
with robust estimates of the data set location and covariance.  The Minimum Covariance Determinant estimator (MCD) has been introduced by P.J.Rousseuw
in [3]_.   Evaluation In this example, we compare the estimation errors that are made when using various types of location and covariance estimates on
contaminated Gaussian distributed data sets:  - The mean and the empirical covariance of the full dataset, which break   down as soon as there are
outliers in the data set - The robust MCD, that has a low error provided   $n_\text{samples} > 5n_\text{features}$ - The mean and the empirical
covariance of the observations that are known   to be good ones. This can be considered as a "perfect" MCD estimation,   so one can trust our
implementation by comparing to this case.    References .. [1] Johanna Hardin, David M Rocke. The distribution of robust distances.     Journal of
Computational and Graphical Statistics. December 1, 2005,     14(4): 928-946. .. [2] Zoubir A., Koivunen V., Chakhchoukh Y. and Muma M. (2012). Robust
estimation in signal processing: A tutorial-style treatment of     fundamental concepts. IEEE Signal Processing Magazine 29(4), 61-80. .. [3] P. J.
Rousseeuw. Least median of squares regression. Journal of American     Statistical Ass., 79:871, 1984.  COMMENT: Authors: The scikit-learn developers
SPDX-License-Identifier: BSD-3-Clause
```python
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
```

### notebooks/dataset2/clustering/plot_feature_agglomeration_vs_univariate_selection.ipynb
CONTEXT: Attempt to remove the temporary cachedir, but don't worry if it fails   COMMENT:
```python
shutil.rmtree(cachedir, ignore_errors=True)
```

### notebooks/dataset2/clustering/plot_feature_agglomeration_vs_univariate_selection.ipynb
CONTEXT: Attempt to remove the temporary cachedir, but don't worry if it fails   COMMENT:
```python
shutil.rmtree(cachedir, ignore_errors=True)
```

### notebooks/dataset2/cross_decomposition/plot_pcr_vs_pls.ipynb
CONTEXT:  Projection on one component and predictive power  We now create two regressors: PCR and PLS, and for our illustration purposes we set the
number of components to 1. Before feeding the data to the PCA step of PCR, we first standardize it, as recommended by good practice. The PLS estimator
has built-in scaling capabilities.  For both models, we plot the projected data onto the first component against the target. In both cases, this
projected data is what the regressors will use as training data.   COMMENT:
```python
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```

## Code Concatenation
```python
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet
shutil.rmtree(cachedir, ignore_errors=True)
shutil.rmtree(cachedir, ignore_errors=True)
print(f"PCR r-squared {pcr.score(X_test, y_test):.3f}")
print(f"PLS r-squared {pls.score(X_test, y_test):.3f}")
```
