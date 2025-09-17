# ================================
# Classificação de Vinhos com KNN (Balanceado)
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import kagglehub  # Certifique-se de estar logado no KaggleHub

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
# 1.3 Mapa de correlação com target
# -------------------------------
corr_target = numeric_df.copy()
corr_target["target"] = (df["quality"] >= 5).astype(int)
plt.figure(figsize=(12,8))
sns.heatmap(corr_target.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.title("Mapa de Correlação das Features com Target")
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

# -------------------------------
# 2b. Pairplot das features principais
# -------------------------------
features_pairplot = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "alcohol", "target"]
sns.pairplot(df_balanced[features_pairplot], hue="target", palette=["darkred","darkgreen"], diag_kind="kde")
plt.suptitle("Pairplot das Features Principais", y=1.02)
plt.show()

# -------------------------------
# 2c. Distribuição KDE por classe
# -------------------------------
plt.figure(figsize=(12,6))
sns.kdeplot(df_balanced[df_balanced["target"]==1]["alcohol"], label="Bom (1)", fill=True, color="darkgreen", alpha=0.5)
sns.kdeplot(df_balanced[df_balanced["target"]==0]["alcohol"], label="Ruim (0)", fill=True, color="darkred", alpha=0.5)
plt.title("Distribuição de Álcool por Classe")
plt.xlabel("Teor de Álcool")
plt.ylabel("Densidade")
plt.legend()
plt.show()

# -------------------------------
# 3. Separar X e y e normalizar
# -------------------------------
X = df_balanced.drop(["quality", "target"], axis=1)
y = df_balanced["target"]
X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Divisão treino/teste
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Tamanho treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# -------------------------------
# 5. Treinamento do Modelo KNN
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

# -------------------------------
# 6. Avaliação
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
# 7. Ajuste do parâmetro k
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
# 8. Visualização da Fronteira de Decisão (t-SNE 2D)
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_tsne, y, test_size=0.3, random_state=42, stratify=y
)

knn2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn2.fit(X_train2, y_train2)
