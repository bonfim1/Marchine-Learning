# =======================================
# Projeto Integrador – Classificação e Clustering no Iris Dataset
# =======================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

# -------------------------------
# Configurações visuais
# -------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
palette = sns.color_palette("Set1")
os.makedirs("figuras", exist_ok=True)

# -------------------------------
# 1. Carregamento e exploração
# -------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("\nInformações gerais:")
print(df.info())
print("\nDescrição estatística:\n", df.describe())
print("\nClasses:\n", df["species"].value_counts())

# Histograma das features
df.hist(figsize=(10, 8), color="skyblue")
plt.suptitle("Distribuição das variáveis")
plt.tight_layout()
plt.savefig("figuras/hist_features.png")
plt.close()

# Mapa de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlação")
plt.savefig("figuras/correlation_map.png")
plt.close()

# -------------------------------
# 2. Pré-processamento
# -------------------------------
X = df.drop("species", axis=1)
y = df["species"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Divisão Treino/Teste
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)
print(f"Tamanho treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# -------------------------------
# 4. Modelos Supervisionados
# -------------------------------
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -------------------------------
# 5. Clustering K-Means
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
sil = silhouette_score(X_scaled, kmeans.labels_)
print("\nSilhouette K-Means:", round(sil, 3))

def best_map(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[l] for l in y_pred])

# -------------------------------
# 6. Avaliação
# -------------------------------
for name, model in {"Decision Tree": dt, "KNN": knn}.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Matriz de Confusão - {name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(f"figuras/cm_{name.replace(' ', '_')}.png")
    plt.close()

# === NOVO BLOCO: Árvore de decisão em imagem ===
plt.figure(figsize=(24,12))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Árvore de Decisão - Iris")
plt.savefig("figuras/arvore_decisao.png", dpi=300, bbox_inches="tight")
plt.close()
print("Árvore de decisão salva em figuras/arvore_decisao.png")

# Avaliação K-Means
y_kmeans_map = best_map(y_enc, kmeans.labels_)
print("\nK-Means (mapeado) Accuracy:", accuracy_score(y_enc, y_kmeans_map))
print(classification_report(y_enc, y_kmeans_map, target_names=le.classes_))
cm_k = confusion_matrix(y_enc, y_kmeans_map)
sns.heatmap(cm_k, annot=True, fmt="d", cmap="Reds",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusão - K-Means")
plt.savefig("figuras/cm_kmeans.png")
plt.close()

# -------------------------------
# 7. Visualizações 2D
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=palette)
plt.title("Distribuição Real (PCA 2D)")
plt.savefig("figuras/pca_real.png")
plt.close()

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=kmeans.labels_, palette=palette)
plt.title("Clusters K-Means (PCA 2D)")
plt.savefig("figuras/pca_kmeans.png")
plt.close()

X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette=palette)
plt.title("Distribuição Real (t-SNE 2D)")
plt.savefig("figuras/tsne_real.png")
plt.close()

sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=kmeans.labels_, palette=palette)
plt.title("Clusters K-Means (t-SNE 2D)")
plt.savefig("figuras/tsne_kmeans.png")
plt.close()

print("\nTodas as figuras foram salvas na pasta 'figuras'.")
