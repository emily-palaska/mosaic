from numpy import mean, std
queries = [
    "SpectralBiclustering with consensus score in scikit-learn",
    "Calibration curves for Naive Bayes and Logistic Regression",
    "Decision boundaries and ellipses for classifiers",
    "Hierarchical clustering with connectivity constraints",
    "Generate blobs with anisotropy and uneven sizes",
    "Decision tree regression with depth comparison",
    "PCA vs KernelPCA image reconstruction"
]
programs = [
    """from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
model.fit(data)

# Compute the similarity of two sets of biclusters
score = consensus_score(
   model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled])
)
print(f"consensus score: {score:.1f}")""",

    """import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

lr = LogisticRegression(C=1.0)
gnb = GaussianNB()
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

clf_list = [
   (lr, "Logistic"),
   (gnb, "Naive Bayes"),
   (gnb_isotonic, "Naive Bayes + Isotonic"),
   (gnb_sigmoid, "Naive Bayes + Sigmoid"),
]
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()""",

    """import matplotlib as mpl
from matplotlib import colors

from sklearn.inspection import DecisionBoundaryDisplay


def plot_ellipse(mean, cov, color, ax):
   v, w = np.linalg.eigh(cov)
   u = w[0] / np.linalg.norm(w[0])
   angle = np.arctan(u[1] / u[0])
   angle = 180 * angle / np.pi  # convert to degrees
   # filled Gaussian at 2 standard deviation
   ell = mpl.patches.Ellipse(
       mean,
       2 * v[0] ** 0.5,
       2 * v[1] ** 0.5,
       angle=180 + angle,
       facecolor=color,
       edgecolor="black",
       linewidth=2,
   )
   ell.set_clip_box(ax.bbox)
   ell.set_alpha(0.4)
   ax.add_artist(ell)


def plot_result(estimator, X, y, ax):
   cmap = colors.ListedColormap(["tab:red", "tab:blue"])
   DecisionBoundaryDisplay.from_estimator(
       estimator,
       X,
       response_method="predict_proba",
       plot_method="pcolormesh",
       ax=ax,
       cmap="RdBu",
       alpha=0.3,
   )
   DecisionBoundaryDisplay.from_estimator(
       estimator,
       X,
       response_method="predict_proba",
       plot_method="contour",
       ax=ax,
       alpha=1.0,
       levels=[0.5],
   )
   y_pred = estimator.predict(X)
   X_right, y_right = X[y == y_pred], y[y == y_pred]
   X_wrong, y_wrong = X[y != y_pred], y[y != y_pred]
   ax.scatter(X_right[:, 0], X_right[:, 1], c=y_right, s=20, cmap=cmap, alpha=0.5)
   ax.scatter(
       X_wrong[:, 0],
       X_wrong[:, 1],
       c=y_wrong,
       s=30,
       cmap=cmap,
       alpha=0.9,
       marker="x",
   )
   ax.scatter(
       estimator.means_[:, 0],
       estimator.means_[:, 1],
       c="yellow",
       s=200,
       marker="*",
       edgecolor="black",
   )

   if isinstance(estimator, LinearDiscriminantAnalysis):
       covariance = [estimator.covariance_] * 2
   else:
       covariance = estimator.covariance_
   plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
   plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)

   ax.set_box_aspect(1)
   ax.spines["top"].set_visible(False)
   ax.spines["bottom"].set_visible(False)
   ax.spines["left"].set_visible(False)
   ax.spines["right"].set_visible(False)
   ax.set(xticks=[], yticks=[])""",

    """import time as time

from sklearn.cluster import AgglomerativeClustering

print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 27  # number of regions
ward = AgglomerativeClustering(
   n_clusters=n_clusters, linkage="ward", connectivity=connectivity
)
ward.fit(X)
label = np.reshape(ward.labels_, rescaled_coins.shape)
print(f"Elapsed time: {time.time() - st:.3f}s")
print(f"Number of pixels: {label.size}")
print(f"Number of clusters: {np.unique(label).size}")

-------------------------------------------- 

import numpy as np

from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
   n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance
X_filtered = np.vstack(
   (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10""",

    """import numpy as np

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()""",

    """from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=32, random_state=42)
kernel_pca = KernelPCA(
   n_components=400,
   kernel="rbf",
   gamma=1e-3,
   fit_inverse_transform=True,
   alpha=5e-3,
   random_state=42,
)

pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(
   kernel_pca.transform(X_test_noisy)
)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))
plot_digits(X_test, "Uncorrupted test images")
plot_digits(
   X_reconstructed_pca,
   f"PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
)
plot_digits(
   X_reconstructed_kernel_pca,
   (
       "Kernel PCA reconstruction\n"
       f"MSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}"
   ),
)"""
]
stopwords = [
    "a", "an", "the",
    "and", "or", "but", "if", "because", "while",
    "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "up", "down", "over", "under", "above", "below",
    "as", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "its", "they", "them", "their",
    "will", "would", "can", "could", "shall", "should",
    "when", "where", "how", "why", "what", "which",
    "not", "no", "yes", "so", "such",
    "there", "here", "other", "another",
    "more", "most", "less", "least",
    "only", "just", "also", "very", "too", "much", "many", "some", "any",
    "each", "every", "all", "both", "either", "neither"
]

program_blocks = [2, 3, 4, 2, 4, 4, 3]
program_lines = [len(program.splitlines()) for program in programs]
filtered_queries = []
for q in queries:
    for word in stopwords:
        q = q.replace(word, '')
    filtered_queries.append(q)
q_words = [len(q.split()) for q in filtered_queries]
assert len(program_blocks) == len(q_words)

program_blocks_norm = [pb / qw for qw, pb in zip(q_words, program_blocks)]
print(f"Ideal Blocks per Query word: {mean(program_blocks_norm)} +- {std(program_blocks_norm)}")
program_lines_norm = [pl / qw for qw, pl in zip(q_words, program_lines)]
print(f"Ideal Lines per Query word: {mean(program_lines_norm)} +- {std(program_lines_norm)}")