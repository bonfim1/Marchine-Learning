# ==========================================================
# Projeto Árvore de Decisão - Previsão de "Skipped" no Spotify
# ==========================================================

# 1️⃣ Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Configurações de visualização
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# ==========================================================
# 2️⃣ Download do dataset do Kaggle
# ==========================================================
# Baixa a última versão do dataset
path = kagglehub.dataset_download("anandshaw2001/top-spotify-songs-in-countries")
print("Path to dataset files:", path)

# Lista os arquivos baixados
print("Arquivos dentro do dataset:")
print(os.listdir(path))

# ==========================================================
# 3️⃣ Carregando os dados
# ==========================================================
# Substitua pelo nome do arquivo CSV que apareceu na listagem
file_path = os.path.join(path, "spotify_history.csv")


# Carrega os dados
df = pd.read_csv(file_path)

# Primeiras linhas
print("Primeiros registros do dataset:")
print(df.head())

# Informações gerais
print("\nInformações gerais:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas das colunas numéricas:")
print(df.describe())

# ==========================================================
# 4️⃣ Escolhendo a variável alvo
# ==========================================================
# Vamos prever se a música será pulada ou não
target_column = "skipped"

# Distribuição da variável alvo
sns.countplot(x=target_column, data=df)
plt.title(f"Distribuição da variável alvo: {target_column}")
plt.show()

# ==========================================================
# 5️⃣ Pré-processamento
# ==========================================================
# 5.1 Verificando valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# 5.2 Removendo valores ausentes
df = df.dropna()

# 5.3 Convertendo colunas categóricas em números (one-hot encoding)
categorical_cols = ["platform", "artist_name", "album_name", "reason_start", "reason_end"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5.4 Separando features e target
X = df.drop(target_column, axis=1)
y = df[target_column]

# ==========================================================
# 6️⃣ Divisão em treino e teste
# ==========================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")

# ==========================================================
# 7️⃣ Treinamento do modelo Decision Tree
# ==========================================================
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ==========================================================
# 8️⃣ Avaliação do modelo
# ==========================================================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Acurácia
print("\nAcurácia do modelo:")
print(accuracy_score(y_test, y_pred))

# Relatório detalhado
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# ==========================================================
# Fim do notebook
# ==========================================================
