# Reverse Embedding Code Synthesis
Query `Denoise handwritten digits with autoencoder methods.`
## Script Variables
- datasets:<br>
>The dataset is a collection of 1797 images of digits. The dataset is divided into 4
- metrics:<br>
>The variable metrics is used to calculate the confusion matrix. It is a matrix that shows the number of
- train_test_split:<br>
>It is a function that splits the dataset into training and testing sets. The test_size parameter specifies the
- svm:<br>
>svm stands for Support Vector Machine. It is a supervised machine learning algorithm that is used for classification and
## Synthesis Blocks
### notebooks/dataset2/classification/plot_digits_classification.ipynb
CONTEXT:   Recognizing hand-written digits  This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.
COMMENT: Import datasets, classifiers and performance metrics
```python
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
```

## Code Concatenation
```python
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
```
