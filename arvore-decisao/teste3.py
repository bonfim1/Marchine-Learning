import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE   # <--- NOVO: para balanceamento

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)
cor_vinho = "darkred"

# ------------------------
# 1. Exploração dos Dados
# ------------------------
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)

# Criar variável alvo binária
df["target"] = (df["quality"] >= 5).astype(int)

print("\nDistribuição original da variável target:")
print(df["target"].value_counts())

# ------------------------
# 2. Pré-processamento
# ------------------------
df = df.dropna()
X = df.drop(["quality", "target"], axis=1)
y = df["target"]

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# 3. Divisão dos Dados
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamanho treino: {X_train.shape[0]} | Tamanho teste: {X_test.shape[0]}")
print("Distribuição antes do balanceamento no treino:\n", y_train.value_counts())

# ------------------------
# Balanceamento (SMOTE)
# ------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nDistribuição após SMOTE no treino:\n", y_train_bal.value_counts())

# ------------------------
# 4. Treinamento
# ------------------------
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train_bal, y_train_bal)

# ------------------------
# 5. Avaliação
# ------------------------
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Importância das variáveis
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

sns.barplot(x=feat_imp, y=feat_imp.index, color=cor_vinho)
plt.title("Importância das Variáveis na Árvore")
plt.show()

# Visualizar a árvore de decisão
plt.figure(figsize=(15,20))
plot_tree(model, feature_names=X.columns, class_names=["ruim","bom"], filled=True)
plt.show()
plt.close()