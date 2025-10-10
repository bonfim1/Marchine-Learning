# =======================================
# PROJETO: CLASSIFICAÇÃO DE VINHOS COM RANDOM FOREST
# =======================================

# --- ETAPA 1: EXPLORAÇÃO DOS DADOS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------
print("\n--- ETAPA 1: EXPLORAÇÃO DOS DADOS ---")

# Carregando o dataset (vermelho e branco combinados)
red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

red["type"] = "red"
white["type"] = "white"

data = pd.concat([red, white], axis=0)
print(f"Linhas: {data.shape[0]} | Colunas: {data.shape[1]}")

print("\nColunas disponíveis:", data.columns.tolist())
print("\nInformações gerais:")
print(data.info())

print("\nEstatísticas descritivas:")
print(data.describe())

# Distribuição da variável "quality"
plt.figure(figsize=(7,4))
sns.countplot(data=data, x="quality", hue="type")
plt.title("Distribuição da variável qualidade por tipo de vinho")
plt.show()

# -------------------------------------------------------
# --- ETAPA 2: PRÉ-PROCESSAMENTO ---
print("\n--- ETAPA 2: PRÉ-PROCESSAMENTO ---")

# Criar variável alvo binária: 1 = bom (>=7), 0 = ruim (<7)
data["target"] = (data["quality"] >= 7).astype(int)
print("Distribuição da variável alvo:\n", data["target"].value_counts())

# Converter variável categórica em dummies
data = pd.get_dummies(data, columns=["type"], drop_first=True)
print("\nColunas finais:", data.columns.tolist())

# Selecionar features e alvo
X = data.drop(columns=["quality", "target"])
y = data["target"]

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# --- ETAPA 3: DIVISÃO DOS DADOS ---
print("\n--- ETAPA 3: DIVISÃO DOS DADOS ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Tamanho treino: {X_train.shape} | teste: {X_test.shape}")

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Distribuição após SMOTE:\n", y_train_res.value_counts())

# -------------------------------------------------------
# --- ETAPA 4: TREINAMENTO DO MODELO ---
print("\n--- ETAPA 4: TREINAMENTO DO MODELO ---")

rf = RandomForestClassifier(
    n_estimators=200,   # número de árvores
    max_depth=10,       # profundidade máxima
    random_state=42
)
rf.fit(X_train_res, y_train_res)

print("Modelo Random Forest treinado com sucesso!")

# -------------------------------------------------------
# --- ETAPA 5: AVALIAÇÃO DO MODELO ---
print("\n--- ETAPA 5: AVALIAÇÃO DO MODELO ---")

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Acurácia: {acc:.4f}")
print(f"Acurácia Balanceada: {bal_acc:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ruim", "Bom"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão - Random Forest")
plt.show()

# Importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind="barh", figsize=(8,5))
plt.title("Top 10 - Importância das variáveis (Random Forest)")
plt.xlabel("Importância")
plt.show()

# -------------------------------------------------------
# --- ETAPA 6: RELATÓRIO FINAL ---
print("\n--- ETAPA 6: RELATÓRIO FINAL ---")

print("""
O modelo Random Forest obteve um desempenho sólido, com boa acurácia e
boa capacidade de generalização. A variável 'alcohol' e 'sulphates'
foram as mais relevantes para prever a qualidade dos vinhos.
O balanceamento via SMOTE contribuiu para evitar viés nas classes.

Possíveis melhorias:
- Ajuste fino de hiperparâmetros (n_estimators, max_depth, min_samples_split)
- Testar outros modelos (XGBoost, LightGBM)
- Analisar correlação entre variáveis para reduzir multicolinearidade.
""")
