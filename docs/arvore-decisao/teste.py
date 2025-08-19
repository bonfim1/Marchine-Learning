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