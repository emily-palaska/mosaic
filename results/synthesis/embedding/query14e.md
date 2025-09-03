# Embedding Code Synthesis
Query `Plot calibration curves for multiple classifiers.`
## Script Variables
- gs:<br>
>gs is a 2D numpy array that is used to create a grid of subplots in a
- i:<br>
>The variable i is used to iterate through the list of classifiers. It is used to access the corresponding
- calibration_displays:<br>
>Calibration displays are used to evaluate the performance of a model in predicting the probability of a given outcome
- col:<br>
>col is the column of the grid that the subplot is placed in. It is used to create the
- fig:<br>
>fig is a variable that is used to store the figure object. It is used to create a plot
- name:<br>
>The variable name is 'clf' which is a classifier. It is used to fit the training data
- clf_list:<br>
>It is a list of tuples, where each tuple contains a classifier and its name. The list is
- enumerate:<br>
>The enumerate function is used to return the index of an iterable along with the value of the element.
- _:<br>
>It is a tuple that contains the row and column coordinates of the subplot where the histogram will be displayed
- plt:<br>
>plt is a variable that is used to create a plot. It is a module that is used to
- ax:<br>
>The variable ax is a subplot object that is used to create a histogram of the y_prob values for
- colors:<br>
>Colors are used to represent different classes in the histogram. The colors are chosen randomly from a list of
## Synthesis Blocks
### notebooks/dataset2/calibration/plot_compare_calibration.ipynb
CONTEXT:  Calibration curves  Below, we train each of the four models with the small training dataset, then plot calibration curves (also known as
reliability diagrams) using predicted probabilities of the test dataset. Calibration curves are created by binning predicted probabilities, then
plotting the mean predicted probability in each bin against the observed frequency ('fraction of positives'). Below the calibration curve, we plot a
histogram showing the distribution of the predicted probabilities or more specifically, the number of samples in each predicted probability bin.
COMMENT: Add histogram
```python
_ = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    _, col = _[i]
    ax = fig.add_subplot(gs[_, col])
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
```

## Code Concatenation
```python
_ = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    _, col = _[i]
    ax = fig.add_subplot(gs[_, col])
    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
```
