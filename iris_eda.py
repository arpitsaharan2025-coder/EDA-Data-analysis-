import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df.drop(columns=["target"], inplace=True)
    df.columns = FEATURES + ["species"]
    return df


def basic_info(df):
    print("=" * 65)
    print("DATASET SHAPE:", df.shape)
    print("=" * 65)
    print(df.head())
    print("=" * 65)
    print(df.dtypes)
    print("=" * 65)
    print("MISSING VALUES:")
    print(df.isnull().sum())
    print("=" * 65)
    print("DUPLICATE ROWS:", df.duplicated().sum())
    print("=" * 65)
    print("CLASS DISTRIBUTION:")
    print(df["species"].value_counts())
    print("=" * 65)
    print("STATISTICAL SUMMARY:")
    print(df.describe())
    print("=" * 65)
    print("STATISTICAL SUMMARY BY SPECIES:")
    print(df.groupby("species")[FEATURES].describe().T)


def variance_range_report(df):
    print("=" * 65)
    print("RANGE AND VARIANCE PER FEATURE:")
    for feature in FEATURES:
        rng = df[feature].max() - df[feature].min()
        var = df[feature].var()
        print(f"{feature}: range={rng:.2f}, variance={var:.3f}")


def outlier_detection(df):
    print("=" * 65)
    print("OUTLIER DETECTION (IQR METHOD):")
    for feature in FEATURES:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[feature] < lower) | (df[feature] > upper)]
        print(f"{feature}: {len(outliers)} outliers detected")


def skewness_kurtosis(df):
    print("=" * 65)
    print("SKEWNESS AND KURTOSIS:")
    for feature in FEATURES:
        print(f"{feature}: skew={df[feature].skew():.3f}, kurtosis={df[feature].kurt():.3f}")


def normality_test(df):
    print("=" * 65)
    print("SHAPIRO-WILK NORMALITY TEST:")
    for feature in FEATURES:
        stat, p = stats.shapiro(df[feature])
        verdict = "Normal" if p > 0.05 else "Not Normal"
        print(f"{feature}: stat={stat:.3f}, p={p:.4f} -> {verdict}")


def anova_test(df):
    print("=" * 65)
    print("ONE-WAY ANOVA ACROSS SPECIES:")
    for feature in FEATURES:
        groups = [df[df["species"] == s][feature] for s in df["species"].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"{feature}: F={f_stat:.3f}, p={p_val:.6f}")


def plot_class_distribution(df):
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="species", data=df, hue="species", legend=False, palette="Set2")
    ax.set_title("Species Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()


def plot_histograms(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.histplot(data=df, x=feature, hue="species", kde=True, ax=axes[i], palette="Set2", element="step")
        axes[i].set_title(f"Distribution of {feature.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig("histograms.png", dpi=150)
    plt.close()


def plot_boxplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.boxplot(data=df, x="species", y=feature, hue="species", legend=False, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{feature.replace('_', ' ').title()} by Species")
    plt.tight_layout()
    plt.savefig("boxplots.png", dpi=150)
    plt.close()


def plot_boxen(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.boxenplot(data=df, x="species", y=feature, hue="species", legend=False, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{feature.replace('_', ' ').title()} Boxen Plot")
    plt.tight_layout()
    plt.savefig("boxen_plots.png", dpi=150)
    plt.close()


def plot_violin(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.violinplot(data=df, x="species", y=feature, hue="species", legend=False, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{feature.replace('_', ' ').title()} Violin Plot")
    plt.tight_layout()
    plt.savefig("violin_plots.png", dpi=150)
    plt.close()


def plot_swarm(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        sns.swarmplot(data=df, x="species", y=feature, hue="species", legend=False, ax=axes[i], palette="Set2", size=4)
        axes[i].set_title(f"{feature.replace('_', ' ').title()} Swarm Plot")
    plt.tight_layout()
    plt.savefig("swarm_plots.png", dpi=150)
    plt.close()


def plot_pairplot(df):
    g = sns.pairplot(df, hue="species", palette="Set2", diag_kind="kde", height=2.2)
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    g.savefig("pairplot.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    corr = df[FEATURES].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()


def plot_scatter_relationships(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", style="species", s=80, ax=axes[0], palette="Set2")
    axes[0].set_title("Sepal Length vs Sepal Width")
    sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", style="species", s=80, ax=axes[1], palette="Set2")
    axes[1].set_title("Petal Length vs Petal Width")
    plt.tight_layout()
    plt.savefig("scatter_relationships.png", dpi=150)
    plt.close()


def plot_jointplot(df):
    g = sns.jointplot(data=df, x="petal_length", y="petal_width", hue="species", palette="Set2", height=7)
    g.fig.suptitle("Petal Length vs Width Joint Distribution", y=1.02)
    g.savefig("jointplot.png", dpi=150)
    plt.close()


def plot_feature_means(df):
    means = df.groupby("species")[FEATURES].mean()
    means.plot(kind="bar", figsize=(10, 6), colormap="Set2")
    plt.title("Mean Feature Values by Species")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=0)
    plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("feature_means.png", dpi=150)
    plt.close()


def plot_parallel_coordinates(df):
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, "species", color=sns.color_palette("Set2", 3))
    plt.title("Parallel Coordinates Plot")
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("parallel_coordinates.png", dpi=150)
    plt.close()


def plot_radar_chart(df):
    means = df.groupby("species")[FEATURES].mean()
    normalized = (means - means.min()) / (means.max() - means.min())
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", len(normalized))
    for i, (species, row) in enumerate(normalized.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=species, color=colors[i], linewidth=2)
        ax.fill(angles, values, color=colors[i], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", " ").title() for f in FEATURES])
    ax.set_title("Normalized Feature Radar Chart", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=150)
    plt.close()


def plot_3d_scatter(df):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"setosa": "#66c2a5", "versicolor": "#fc8d62", "virginica": "#8da0cb"}
    for species, group in df.groupby("species"):
        ax.scatter(group["sepal_length"], group["sepal_width"], group["petal_length"],
                   label=species, color=colors[species], s=50, alpha=0.8)
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    ax.set_title("3D Feature Space")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scatter_3d.png", dpi=150)
    plt.close()


def plot_pca(df):
    x = df[FEATURES].values
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["species"] = df["species"].values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="species", style="species", s=100, palette="Set2")
    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    plt.title(f"PCA Projection (Total Variance: {var1 + var2:.2f}%)")
    plt.xlabel(f"PC1 ({var1:.2f}%)")
    plt.ylabel(f"PC2 ({var2:.2f}%)")
    plt.tight_layout()
    plt.savefig("pca_projection.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    basic_info(df)
    variance_range_report(df)
    outlier_detection(df)
    skewness_kurtosis(df)
    normality_test(df)
    anova_test(df)
    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_boxen(df)
    plot_violin(df)
    plot_swarm(df)
    plot_pairplot(df)
    plot_correlation_heatmap(df)
    plot_scatter_relationships(df)
    plot_jointplot(df)
    plot_feature_means(df)
    plot_parallel_coordinates(df)
    plot_radar_chart(df)
    plot_3d_scatter(df)
    plot_pca(df)
    print("=" * 65)
    print("EDA COMPLETE. ALL PLOTS SAVED TO CURRENT DIRECTORY.")


if __name__ == "__main__":
    main()
