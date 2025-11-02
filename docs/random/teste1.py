# ============================================================
# PROJETO: CLASSIFICAÇÃO DE VINHOS COM RANDOM FOREST
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Definir paleta vinho
paleta_vinho = ["#7B1113", "#A02C2D", "#C44B4D", "#D67B7C", "#F2A6A6"]

# ============================================================
# === ETAPA 1: EXPLORAÇÃO DOS DADOS ===
print("\n--- ETAPA 1: EXPLORAÇÃO DOS DADOS ---")

print("Baixando dataset...")
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)
print("Dataset carregado!\n")

print(df.info())
print(df.describe())

# Distribuição da qualidade
plt.figure(figsize=(7,4))
sns.countplot(data=df, x="quality", hue="type", palette=paleta_vinho)
plt.title("Distribuição da qualidade por tipo de vinho")
plt.show()

# ============================================================
# === ETAPA 2: PRÉ-PROCESSAMENTO ===
print("\n--- ETAPA 2: PRÉ-PROCESSAMENTO ---")

def classificar_qualidade(valor):
    if valor <= 5:
        return 'ruim'
    elif valor == 6:
        return 'médio'
    else:
        return 'bom'

df['categoria'] = df['quality'].apply(classificar_qualidade)
df = pd.get_dummies(df, columns=['type'], drop_first=True)

X = df.drop(columns=['quality', 'categoria'])
y = df['categoria']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# === ETAPA 3: DIVISÃO DOS DADOS ===
print("\n--- ETAPA 3: DIVISÃO DOS DADOS ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# === ETAPA 4: TREINAMENTO ===
print("\n--- ETAPA 4: TREINAMENTO ---")

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
modelo.fit(X_train, y_train)

# ============================================================
# === ETAPA 5: AVALIAÇÃO ===
print("\n--- ETAPA 5: AVALIAÇÃO ---")

y_pred = modelo.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.3f}")

# 1️⃣ Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Reds", fmt='d', cbar=False,
            xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# 2️⃣ Importância das variáveis
importances = pd.Series(modelo.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
importances.sort_values().tail(10).plot(kind="barh", color="#7B1113")
plt.title("Top 10 - Importância das Variáveis")
plt.show()

# 3️⃣ Distribuição real vs prevista
fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.countplot(y_test, ax=ax[0], palette=paleta_vinho)
ax[0].set_title("Distribuição Real")
sns.countplot(y_pred, ax=ax[1], palette=paleta_vinho)
ax[1].set_title("Distribuição Prevista")
plt.show()

# 4️⃣ Curva de aprendizado
train_sizes, train_scores, test_scores = learning_curve(
    modelo, X_scaled, y, cv=5, scoring='accuracy'
)
plt.figure(figsize=(7,4))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Treino", color="#7B1113")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validação", color="#A02C2D")
plt.title("Curva de Aprendizado - Random Forest")
plt.xlabel("Tamanho do Conjunto de Treino")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

# 5️⃣ Visualização de duas árvores do Random Forest 🌳
estimator1 = modelo.estimators_[0]
estimator2 = modelo.estimators_[1]

plt.figure(figsize=(20,10))
plot_tree(estimator1, filled=True, feature_names=X.columns,
          class_names=modelo.classes_, max_depth=3, fontsize=10)
plt.title("Árvore 1 - Random Forest")
plt.show()

plt.figure(figsize=(20,10))
plot_tree(estimator2, filled=True, feature_names=X.columns,
          class_names=modelo.classes_, max_depth=3, fontsize=10)
plt.title("Árvore 2 - Random Forest")
plt.show()


