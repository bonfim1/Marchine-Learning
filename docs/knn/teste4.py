# ================================
# Classificação de Vinhos com KNN (Balanceado)
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample  # Para balanceamento

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
# Variável alvo binária: 1 = bom (quality >=5), 0 = ruim
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
# 4. Treinamento do Modelo KNN
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

# -------------------------------
# 5. Avaliação
# -------------------------------
y_pred = knn.predict(X_test)
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão - KNN (Balanceado)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# -------------------------------
# 6. Ajuste do parâmetro k
# -------------------------------
scores = []
for k in range(1, 21):
    modelo = KNeighborsClassifier(n_neighbors=k, weights='distance')
    modelo.fit(X_train, y_train)
    scores.append(modelo.score(X_test, y_test))

plt.figure(figsize=(12,5))
plt.plot(range(1,21), scores, marker="o", color=cor_vinho)
plt.title("Variação da Acurácia com k (Balanceado)")
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia no teste")
plt.show()

# -------------------------------
# 7. Visualização da Fronteira de Decisão (t-SNE 2D)
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_tsne, y, test_size=0.3, random_state=42, stratify=y
)

knn2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn2.fit(X_train2, y_train2)

x_min, x_max = X_tsne[:, 0].min() - 5, X_tsne[:, 0].max() + 5
y_min, y_max = X_tsne[:, 1].min() - 5, X_tsne[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["darkred", "darkgreen"]

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train2[:, 0], y=X_train2[:, 1], hue=y_train2,
    palette=cmap_bold, edgecolor="k", s=20, alpha=0.6
)

plt.title("KNN Decision Boundary (t-SNE 2D, Balanceado)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
plt.close()
#usar esse
