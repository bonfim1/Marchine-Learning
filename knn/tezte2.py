# ================================
# Classificação de Vinhos com KNN (Balanceado + Fronteira limpa)
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import umap  # UMAP para projeção 2D limpa

# -------------------------------
# 1. Exploração dos Dados
# -------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)
cor_vinho = "darkred"

# Baixar dataset
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
print("Path to dataset files:", path)
print("Arquivos:", os.listdir(path))

# Carregar CSV
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)
print("\nPrimeiros registros:\n", df.head())
print("\nInformações gerais:")
print(df.info())
print("\nDistribuição quality:\n", df["quality"].value_counts())
print("\nEstatísticas:\n", df.describe())

# Histograma da qualidade
sns.histplot(df["quality"], bins=7, kde=True, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.show()

# Mapa de correlação
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap="Reds")
plt.title("Mapa de Correlação")
plt.show()

# -------------------------------
# 2. Pré-processamento
# -------------------------------

# Variável alvo binária: 1 = bom (quality >=5), 0 = ruim
df["target"] = (df["quality"] >= 5).astype(int)
print("\nDistribuição target original:\n", df["target"].value_counts())

# Remover ausentes
df = df.dropna()

# -------------------------------
# 2a. Balanceamento com Oversampling da classe minoritária
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

# Dummies para variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Divisão dos Dados
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

# =========================================
# 7. Visualização da Fronteira de Decisão (UMAP 2D)
# =========================================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Divisão treino/teste
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_umap, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar KNN em 2D UMAP
knn2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn2.fit(X_train2, y_train2)


# Criar malha fina para fronteira
x_min, x_max = X_umap[:, 0].min() - 0.5, X_umap[:, 0].max() + 0.5
y_min, y_max = X_umap[:, 1].min() - 0.5, X_umap[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
                

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ["darkred", "darkgreen"]

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train2[:, 0], y=X_train2[:, 1], hue=y_train2,
    palette=cmap_bold, edgecolor="k", s=30, alpha=0.8
)

plt.title("KNN Decision Boundary (UMAP 2D, Balanceado)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()
plt.close()