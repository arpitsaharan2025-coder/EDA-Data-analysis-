import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_mnist():
    try:
        bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(int)
        dim = (28, 28)
    except Exception as e:
        print(f"OpenML unreachable ({e}); using sklearn's digits dataset as fallback.")
        bunch = load_digits()
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(int)
        dim = (8, 8)
    return X, y, dim


def subsample(X, y, size, seed=42):
    if size >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
    return Xs, ys


def evaluate_knn(Xtr, Xte, ytr, yte, label):
    clf = KNeighborsClassifier(n_neighbors=5)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    train_time = time.time() - t0
    t0 = time.time()
    preds = clf.predict(Xte)
    predict_time = time.time() - t0
    acc = accuracy_score(yte, preds)
    print(f"{label}: accuracy={acc:.4f}, train_time={train_time:.4f}s, predict_time={predict_time:.4f}s")
    return acc, train_time, predict_time


X, y, IMG_DIM = load_mnist()
N_SAMPLES, N_FEATURES = X.shape
N_CLASSES = len(np.unique(y))

print("Part A: Data Understanding")
print(f"Number of samples: {N_SAMPLES}")
print(f"Number of classes: {N_CLASSES}")
print(f"Image dimensions: {IMG_DIM[0]} x {IMG_DIM[1]}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
rng = np.random.RandomState(0)
sample_idx = rng.choice(N_SAMPLES, 10, replace=False)
for ax, idx in zip(axes.flat, sample_idx):
    ax.imshow(X[idx].reshape(IMG_DIM), cmap="gray")
    ax.set_title(str(y[idx]))
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_sample_digits.png", dpi=130)
plt.close()

X = X / X.max()

SAMPLE_SIZE = min(10000, N_SAMPLES)
X_sub, y_sub = subsample(X, y, SAMPLE_SIZE)
print(f"Using a stratified subsample of {len(X_sub)} images for PCA/t-SNE/classification.")

print("\nPart B: PCA")
pca_full = PCA(random_state=42).fit(X_sub)
var_ratio = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)


def components_for(threshold):
    return int(np.argmax(cum_var >= threshold) + 1)


n90, n95, n99 = components_for(0.90), components_for(0.95), components_for(0.99)
print(f"Components for 90% variance: {n90}")
print(f"Components for 95% variance: {n95}")
print(f"Components for 99% variance: {n99}")

pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_sub)

N_PCA50 = min(50, N_FEATURES)
pca_50d = PCA(n_components=N_PCA50, random_state=42)
X_pca_50d = pca_50d.fit_transform(X_sub)

n_scree = min(50, len(var_ratio))
plt.figure(figsize=(7, 5))
plt.plot(range(1, n_scree + 1), var_ratio[:n_scree], marker="o", markersize=3)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_scree_plot.png", dpi=130)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var)
for thresh, n_c in zip([0.90, 0.95, 0.99], [n90, n95, n99]):
    plt.axhline(thresh, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(n_c, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_cumulative_variance.png", dpi=130)
plt.close()

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_sub, cmap="tab10", s=6, alpha=0.6)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_pca_2d_projection.png", dpi=130)
plt.close()

print("\nPart C: t-SNE")
TSNE_SIZE = min(2000, len(X_sub))
X_tsne_sub, y_tsne_sub = subsample(X_sub, y_sub, TSNE_SIZE)
X_pca50_tsne_sub = pca_50d.transform(X_tsne_sub)

perplexities = [5, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sc = None
for col, perp in enumerate(perplexities):
    emb_orig = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(X_tsne_sub)
    axes[0, col].scatter(emb_orig[:, 0], emb_orig[:, 1], c=y_tsne_sub, cmap="tab10", s=6, alpha=0.7)
    axes[0, col].set_title(f"Original data, perplexity={perp}")

    emb_pca = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(X_pca50_tsne_sub)
    sc = axes[1, col].scatter(emb_pca[:, 0], emb_pca[:, 1], c=y_tsne_sub, cmap="tab10", s=6, alpha=0.7)
    axes[1, col].set_title(f"PCA-{N_PCA50}D data, perplexity={perp}")

fig.colorbar(sc, ax=axes.ravel().tolist(), label="Digit", fraction=0.02)
plt.savefig(f"{OUT_DIR}/05_tsne_comparison.png", dpi=130)
plt.close()

print("\nPart D: Digit Recognition")
Xtr, Xte, ytr, yte = train_test_split(X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42)

pca_clf = PCA(n_components=N_PCA50, random_state=42).fit(Xtr)
Xtr_pca = pca_clf.transform(Xtr)
Xte_pca = pca_clf.transform(Xte)

res_orig = evaluate_knn(Xtr, Xte, ytr, yte, f"KNN on Original ({N_FEATURES}D)")
res_pca = evaluate_knn(Xtr_pca, Xte_pca, ytr, yte, f"KNN on PCA ({N_PCA50}D)")

with open(f"{OUT_DIR}/results_summary.txt", "w") as f:
    f.write(f"Samples: {N_SAMPLES}\nClasses: {N_CLASSES}\nImage dimensions: {IMG_DIM}\n\n")
    f.write(f"Components for 90% variance: {n90}\n")
    f.write(f"Components for 95% variance: {n95}\n")
    f.write(f"Components for 99% variance: {n99}\n\n")
    f.write(f"KNN Original ({N_FEATURES}D): accuracy={res_orig[0]:.4f}, train_time={res_orig[1]:.4f}s, predict_time={res_orig[2]:.4f}s\n")
    f.write(f"KNN PCA ({N_PCA50}D): accuracy={res_pca[0]:.4f}, train_time={res_pca[1]:.4f}s, predict_time={res_pca[2]:.4f}s\n")
    f.write(f"Speedup (train): {res_orig[1]/res_pca[1]:.2f}x\n")
    f.write(f"Speedup (predict): {res_orig[2]/res_pca[2]:.2f}x\n")

print("\nAll plots and results saved in the outputs/ directory.")
