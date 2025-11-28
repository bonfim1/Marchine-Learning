import pandas as pd                 # Manipulação de dados em tabelas (DataFrames)
import numpy as np                  # Operações numéricas e matrizes
import matplotlib.pyplot as plt     # Criação de gráficos
import seaborn as sns               # Gráficos mais bonitos e estilizados
import kagglehub                    # Para baixar datasets do Kaggle
import os                           # Para manipulação de caminhos de arquivos

from sklearn.model_selection import train_test_split            # Separar dados em treino e teste
from sklearn.tree import DecisionTreeClassifier, plot_tree      # Modelo de árvore de decisão e plotagem
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)  # Métricas de avaliação
from sklearn.preprocessing import StandardScaler                # Normalização dos dados


sns.set_style("whitegrid")                # Estilo de fundo dos gráficos
plt.rcParams["figure.figsize"] = (10,6)   # Tamanho padrão dos gráficos
cor_vinho = "darkred"                     # Cor padrão para gráficos

# 1. Exploração dos Dados

# Baixar dataset do Kaggle
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
print("Path to dataset files:", path)
print("\nArquivos dentro do dataset:")
print(os.listdir(path))                   # Listar arquivos baixados

# Usar arquivo CSV específico
file_path = os.path.join(path, "wine_quality_merged.csv")

# Carregar dataset em um DataFrame
df = pd.read_csv(file_path, index_col=0)  # index_col=0 define a primeira coluna como índice
print("\nPrimeiros registros:")
print(df.head())                          # Mostrar as 5 primeiras linhas

print("\nInformações gerais:")
print(df.info())                           # Informações sobre colunas, tipos e valores nulos

print("\nDistribuição da variável quality original:")
print(df["quality"].value_counts().sort_index())  # Contagem de cada valor de quality

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())                       # Média, desvio padrão, min, max, quartis

# Visualização da distribuição da qualidade
sns.histplot(df["quality"], bins=7, kde=True, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.show()

# Mapa de correlação entre colunas numéricas
numeric_df = df.select_dtypes(include=np.number)   # Seleciona apenas colunas numéricas
sns.heatmap(numeric_df.corr(), cmap="Reds", annot=False)
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.show()


# 2. Pré-processamento


# Criar variável alvo binária: bom vinho (1) ou ruim vinho (0)
df["target"] = (df["quality"] >= 5).astype(int)

print("\nDistribuição após transformação para target:")
print(df["target"].value_counts())

# Gráfico da variável alvo
sns.countplot(x="target", data=df, color=cor_vinho)
plt.title("Distribuição da variável alvo (0=ruim, 1=bom)")
plt.show()

# Verificar valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum())

# Remover linhas com valores ausentes
df = df.dropna()

# Separar variáveis explicativas (X) e variável alvo (y)
X = df.drop(["quality", "target"], axis=1)  # Todas as colunas menos quality e target
y = df["target"]                             # Apenas a coluna target

# Transformar variáveis categóricas em variáveis dummy (0 ou 1)
X = pd.get_dummies(X, drop_first=True)       # drop_first=True evita multicolinearidade

# Normalizar os dados numéricos
scaler = StandardScaler()                    # Cria o objeto para padronização
X_scaled = scaler.fit_transform(X)           # Calcula média e desvio e transforma os dados


# 3. Divisão dos Dados


# Separar em treino (70%) e teste (30%), mantendo proporção da classe (stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")

# 4. Treinamento do Modelo


# Criar modelo de Árvore de Decisão com profundidade máxima de 5
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)                  # Treinar modelo nos dados de treino


# 5. Avaliação do Modelo

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Métricas principais
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Métricas individuais
print("Precisão:", precision_score(y_test, y_pred))
print("Revocação (Recall):", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Importância das variáveis
importances = model.feature_importances_                # Extrai importância de cada feature
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Gráfico das variáveis mais importantes
sns.barplot(x=feat_imp, y=feat_imp.index, color=cor_vinho)
plt.title("Importância das Variáveis na Árvore")
plt.show()

# Visualizar a árvore de decisão
plt.figure(figsize=(15,20))
plot_tree(model, feature_names=X.columns, class_names=["ruim","bom"], filled=True)
plt.show()
plt.close()