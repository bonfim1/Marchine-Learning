import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# Baixar dataset
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
print("Path to dataset files:", path)

print("\nArquivos dentro do dataset:")
print(os.listdir(path))

# Usar o arquivo cleaned
file_path = os.path.join(path, "wine_quality_merged.csv")

# Carregar dataset
df = pd.read_csv(file_path, index_col=0)
print("\nPrimeiros registros:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nDistribuição da variável quality original:")
print(df["quality"].value_counts().sort_index())

# ======================
# Criar variável alvo (target)
# 1 = bom (quality >= 6), 0 = ruim (quality < 6)
# ======================
df["target"] = (df["quality"] >= 5).astype(int)

print("\nDistribuição após transformação para target:")
print(df["target"].value_counts())

sns.countplot(x="target", data=df)
plt.title("Distribuição da variável alvo (0=ruim, 1=bom)")
plt.show()

# ======================
# Pré-processamento
# ======================
print("\nValores ausentes:")
print(df.isnull().sum())

# Features: tirando quality e target
X = df.drop(["quality", "target"], axis=1)
y = df["target"]

# Se houver variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")

# ======================
# Modelo Árvore de Decisão
# ======================
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Visualizar a árvore
plt.figure(figsize=(18,10))
plot_tree(model, feature_names=X.columns, class_names=["ruim","bom"], filled=True)
plt.show()
