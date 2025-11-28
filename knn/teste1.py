# ================================
# Exercício - Classificação com KNN
# ================================

# 1. Bibliotecas
import pandas as pd                 # Manipulação de dados em tabelas (DataFrames)
import numpy as np                  # Operações numéricas e matrizes
import matplotlib.pyplot as plt     # Criação de gráficos
import seaborn as sns               # Gráficos mais bonitos e estilizados
import kagglehub                    # Para baixar datasets do Kaggle
import os                           # Para manipulação de caminhos de arquivos

# Scikit-learn (machine learning)
from sklearn.model_selection import train_test_split   # Separar dados em treino e teste
from sklearn.preprocessing import StandardScaler       # Normalização dos dados
from sklearn.neighbors import KNeighborsClassifier     # Algoritmo KNN
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)         # Métricas de avaliação


# Configurações de estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)
cor_vinho = "darkred"   # Cor padrão dos gráficos


# ======================================
# 2. Exploração dos Dados
# ======================================

# Baixar dataset do Kaggle (será salvo localmente na pasta .kagglehub)
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
print("Path to dataset files:", path)
print("\nArquivos dentro do dataset:")
print(os.listdir(path))  # Listar arquivos disponíveis

# Carregar o arquivo CSV principal
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)  # index_col=0 define a primeira coluna como índice

# Mostrar primeiras linhas
print("\nPrimeiros registros:")
print(df.head())

# Informações gerais
print("\nInformações gerais:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Distribuição da variável "quality" original
print("\nDistribuição da variável quality:")
print(df["quality"].value_counts().sort_index())

# Gráfico da qualidade
sns.histplot(df["quality"], bins=7, kde=True, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.show()

# Mapa de correlação entre variáveis numéricas
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap="Reds")
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.show()


# ======================================
# 3. Pré-processamento
# ======================================

# Criar variável alvo binária:
# Bom vinho (1) se quality >= 5, caso contrário Ruim (0)
df["target"] = (df["quality"] >= 5).astype(int)

# Distribuição da variável alvo
print("\nDistribuição da variável alvo (target):")
print(df["target"].value_counts())

# Gráfico da variável alvo
sns.countplot(x="target", data=df, color=cor_vinho)
plt.title("Distribuição da variável alvo (0=ruim, 1=bom)")
plt.show()

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Remover registros com valores ausentes (se houver)
df = df.dropna()

# Separar variáveis explicativas (X) e variável alvo (y)
X = df.drop(["quality", "target"], axis=1)   # Todas menos qualidade e alvo
y = df["target"]

# Transformar variáveis categóricas em dummies (0/1)
X = pd.get_dummies(X, drop_first=True)

# Normalizar dados (KNN depende de distância → normalização é fundamental)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ======================================
# 4. Divisão dos Dados
# ======================================

# Dividir em treino (70%) e teste (30%), mantendo proporção da classe
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")


# ======================================
# 5. Treinamento do Modelo (KNN)
# ======================================

# Criar modelo KNN com k=5 vizinhos
model = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo nos dados de treino
model.fit(X_train, y_train)


# ======================================
# 6. Avaliação do Modelo
# ======================================

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Acurácia
print("\nAcurácia:", accuracy_score(y_test, y_pred))

# Relatório de classificação (precisão, recall, f1-score)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()


# ======================================
# 7. Testando diferentes valores de k (extra)
# ======================================

scores = []
for k in range(1, 21):   # Testar k de 1 até 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# Plotar gráfico de acurácia vs. k
plt.plot(range(1, 21), scores, marker="o", color=cor_vinho)
plt.xlabel("Número de vizinhos (k)")
plt.ylabel("Acurácia no teste")
plt.title("Variação da Acurácia com k")
plt.show()
plt.close()