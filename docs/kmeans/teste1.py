# ================================
# Classificação de Vinhos com K-Means (Balanceado)
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
from sklearn.utils import resample
from scipy.optimize import linear_sum_assignment

# -------------------------------
# Configurações visuais
# -------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)
cor_vinho = "darkred"
sns.set_palette(sns.color_palette([cor_vinho]))

# -------------------------------
# 1. Download e leitura do dataset
# -------------------------------
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)

# Exploração inicial
print("\nPrimeiros registros:\n", df.head())
print("\nInformações gerais:")
print(df.info())
print("\nDistribuição quality:\n", df["quality"].value_counts())
print("\nEstatísticas:\n", df.describe())

# -------------------------------
# 1.1 Histograma da qualidade
# -------------------------------
sns.histplot(df["quality"], bins=7, kde=True, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.xlabel("Quality")
plt.ylabel("Contagem")
plt.show()

# -------------------------------
# 1.2 Mapa de correlação
# -------------------------------
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.title("Mapa de Correlação")
plt.show()

# -------------------------------
# 2. Pré-processamento
# -------------------------------
df["target"] = (df["quality"] >= 5).astype(int)
df = df.dropna()
print("\nDistribuição target original:\n", df["target"].value_counts())

# -------------------------------
# 2a. Balanceamento (Oversampling)
# -------------------------------
df_majority = df[df.target==1]
df_minority = df[df.target==0]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle
print("\nDistribuição target após balanceamento:\n", df_balanced["target"].value_counts())

# Separar X e y
X = df_balanced.drop(["quality", "target"], axis=1)
y = df_balanced["target"]
X = pd.get_dummies(X, drop_first=True)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Divisão treino/teste
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Tamanho treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# -------------------------------
# 4. Treinamento do Modelo K-Means
# -------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# -------------------------------
# 5. Avaliação
# -------------------------------
y_pred = kmeans.predict(X_test)

# Função para mapear clusters para classes reais
def best_map(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(D):
        for j in range(D):
            cost_matrix[i, j] = np.sum((y_true == i) & (y_pred == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping[label] for label in y_pred])

y_pred_mapped = best_map(y_test, y_pred)

print("\nAcurácia:", accuracy_score(y_test, y_pred_mapped))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_mapped))
print("Adjusted Rand Index:", adjusted_rand_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred_mapped))

# Matriz de confusão visual
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_mapped), annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão - K-Means (Balanceado)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# -------------------------------
# 6. Testando diferentes clusters
# -------------------------------
scores = []
for k in range(2, 10):
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo.fit(X_train)
    pred = modelo.predict(X_test)
    pred_mapped = best_map(y_test, pred)
    scores.append(accuracy_score(y_test, pred_mapped))

plt.figure(figsize=(12,5))
plt.plot(range(2,10), scores, marker="o", color=cor_vinho)
plt.title("Variação da Acurácia com n_clusters (Balanceado)")
plt.xlabel("Número de clusters")
plt.ylabel("Acurácia no teste")
plt.show()

# -------------------------------
# 7. Visualização em 2D (t-SNE)
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10,8))
sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y,
    palette=["darkred","darkgreen"], alpha=0.6, edgecolor="k"
)
plt.title("Distribuição Real (t-SNE 2D)")
plt.show()

clusters = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_scaled)

plt.figure(figsize=(10,8))
sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters,
    palette=["darkred","darkgreen"], alpha=0.6, edgecolor="k"
)
plt.title("Clusters K-Means (t-SNE 2D)")
plt.show()
plt.close()
