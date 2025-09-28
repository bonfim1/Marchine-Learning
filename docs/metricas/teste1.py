import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, balanced_accuracy_score, f1_score,
    silhouette_score
)
from imblearn.over_sampling import SMOTE

# --------------------------
# Configurações visuais
# --------------------------
sns.set_style("whitegrid")
sns.set_palette("viridis")

# ============================================================
# ETAPA 1 – EXPLORAÇÃO DOS DADOS
# ============================================================
print("\n--- ETAPA 1: EXPLORAÇÃO DOS DADOS ---")

print("Baixando dataset...")
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)
print("Dataset carregado!\n")

print("Colunas disponíveis:", df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())
print("\nInformações gerais:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())

# Distribuição da qualidade
plt.figure(figsize=(8,5))
sns.countplot(x="quality", data=df)
plt.title("Distribuição da Qualidade Original")
plt.show()

# Mapa de correlação
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f")
plt.title("Mapa de Correlação")
plt.show()

# ============================================================
# ETAPA 2 – PRÉ-PROCESSAMENTO
# ============================================================
print("\n--- ETAPA 2: PRÉ-PROCESSAMENTO ---")

# Limpeza de valores ausentes
df.dropna(inplace=True)

# Variável alvo binária: bom (>=6) ou ruim (<6)
df["target"] = (df["quality"] >= 6).astype(int)
print("\nDistribuição de 'target':")
print(df["target"].value_counts())

# Separação X e y (mantendo a coluna de tipo se existir)
X = df.drop(["quality", "target"], axis=1)
y = df["target"]

# One-hot encoding se houver coluna de tipo de vinho
for possible_col in ["wine_type", "type"]:
    if possible_col in X.columns:
        X = pd.get_dummies(X, columns=[possible_col], drop_first=True)
        print(f"Coluna categórica '{possible_col}' convertida em dummies.")
        break

print("\nColunas após pré-processamento:", X.columns.tolist())

# ============================================================
# ETAPA 3 – DIVISÃO DOS DADOS
# ============================================================
print("\n--- ETAPA 3: DIVISÃO DOS DADOS ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Tamanho treino:", X_train.shape, "| teste:", X_test.shape)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
print("Distribuição após SMOTE:", y_train_bal.value_counts())

# ============================================================
# ETAPA 4 – TREINAMENTO KNN
# ============================================================
print("\n--- ETAPA 4: TREINAMENTO KNN ---")
k = 7
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_bal, y_train_bal)
print(f"Modelo KNN treinado com k={k}")

# ============================================================
# ETAPA 5 – AVALIAÇÃO KNN
# ============================================================
print("\n--- ETAPA 5: AVALIAÇÃO KNN ---")
y_pred = knn_model.predict(X_test_scaled)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Acurácia Balanceada:", balanced_accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Ruim", "Bom"]).plot(cmap="viridis")
plt.title("Matriz de Confusão – KNN")
plt.show()

# Ajuste de hiperparâmetro K
k_range = range(1, 30, 2)
f1_scores = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_bal, y_train_bal)
    pred_i = knn.predict(X_test_scaled)
    f1_scores.append(f1_score(y_test, pred_i, average="weighted"))
best_k = k_range[np.argmax(f1_scores)]
print("Melhor k baseado em F1:", best_k)

plt.figure(figsize=(10,5))
plt.plot(k_range, f1_scores, marker="o")
plt.title("F1-Score vs k")
plt.xlabel("k")
plt.ylabel("F1-Score")
plt.axvline(best_k, color="red", linestyle="--")
plt.show()

# ============================================================
# ETAPA 6 – CLUSTERING K-MEANS
# ============================================================
print("\n--- ETAPA 6: CLUSTERING K-MEANS ---")
# Usamos as mesmas features normalizadas (sem y)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))
df["kmeans_cluster"] = clusters

print("Silhouette Score:", silhouette_score(StandardScaler().fit_transform(X), clusters))
print("Centroides:\n", kmeans.cluster_centers_)

plt.figure(figsize=(8,5))
sns.countplot(x="kmeans_cluster", data=df)
plt.title("Distribuição dos Clusters K-Means")
plt.show()
