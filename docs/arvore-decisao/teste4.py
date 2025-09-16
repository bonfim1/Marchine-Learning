# teste4_completo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

# --- Configurações visuais ---
sns.set_style("whitegrid")
cor_vinho = "darkred"
sns.set_palette(sns.color_palette([cor_vinho]))

# --- 1. Download e leitura do dataset ---
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)

# --- 1.1 Exploração inicial ---
print("\nPrimeiros registros:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nDistribuição da variável quality original:")
print(df["quality"].value_counts().sort_index())

print("\nEstatísticas descritivas:")
print(df.describe())

# --- 1.2 Distribuição da qualidade do vinho ---
plt.figure(figsize=(10,6))
sns.countplot(x="quality", data=df, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.xlabel("Quality")
plt.ylabel("Contagem")
plt.show()

# --- 1.3 Mapa de correlação numérica ---
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.show()

# --- 2. Pré-processamento ---
df = df.dropna()
df["target"] = (df["quality"] >= 5).astype(int)

print("\nDistribuição original da variável target:")
print(df["target"].value_counts())

X = df.drop(["quality", "target"], axis=1)
y = df["target"]

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Divisão treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamanho treino: {X_train.shape[0]} | Tamanho teste: {X_test.shape[0]}")
print("Distribuição antes do balanceamento no treino:\n", y_train.value_counts())

# --- 4. Balanceamento com SMOTE ---
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nDistribuição após SMOTE no treino:\n", y_train_bal.value_counts())

# --- 5. Treinamento ---
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train_bal, y_train_bal)

# --- 6. Avaliação ---
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# --- 7. Gráficos ---

# 7.1 Distribuição da variável target após SMOTE
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_bal, color=cor_vinho)
plt.title("Distribuição da variável alvo após SMOTE (treino)")
plt.xlabel("Target (0=ruim, 1=bom)")
plt.ylabel("Quantidade")
plt.show()

# 7.2 Importância das variáveis
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp, y=feat_imp.index, color=cor_vinho)
plt.title("Importância das Variáveis na Árvore de Decisão")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.show()

# 7.3 Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# 7.4 Árvore de Decisão (matplotlib ajustável)
num_features = len(X.columns)
tree_depth = model.get_depth()
num_nodes = model.tree_.node_count

fig_width = max(12, num_features * 1.2)
fig_height = max(8, tree_depth * 1.5)
plt.figure(figsize=(fig_width, fig_height))

fontsize = min(10, max(5, 80 / num_nodes))

plot_tree(   # <-- aqui ajustei para usar plot_tree em vez de tree.plot_tree
    model,
    feature_names=X.columns,
    class_names=["ruim", "bom"],
    filled=True,
    rounded=True,
    fontsize=fontsize
)
plt.tight_layout()
plt.savefig("arvore_decisao_matplotlib.pdf", format="pdf", bbox_inches="tight")
plt.show()
