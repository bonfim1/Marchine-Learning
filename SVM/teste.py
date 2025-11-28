# ============================
# 1. IMPORTS
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import kagglehub
import seaborn as sns
import os

# ============================
# 2. CARREGAR O DATASET
# ============================
print("Baixando dataset...")
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)
print("Dataset carregado!\n")

# ============================
# 3. EXPLORAÇÃO DOS DADOS
# ============================
print(df.info())
print(df.describe())

plt.figure(figsize=(10,6))
sns.countplot(data=df, x="quality")
plt.title("Distribuição das notas de qualidade")
plt.show()

sns.pairplot(df.iloc[:, :5])
plt.show()

# ============================
# 4. PRÉ-PROCESSAMENTO
# ============================

# Transformar quality em classe (classificação)
def classificar(q):
    if q <= 5:
        return "ruim"
    elif q == 6:
        return "medio"
    else:
        return "bom"

df["label"] = df["quality"].apply(classificar)

# Remover coluna original
X = df.drop(columns=["quality", "label"])
y = df["label"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# 5. DIVISÃO TREINO/TESTE
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# ============================
# 6. TREINAMENTO DO MODELO SVM
# ============================
model = SVC(kernel="rbf", C=2, gamma="scale")
model.fit(X_train, y_train)

# ============================
# 7. AVALIAÇÃO
# ============================
y_pred = model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Matriz de Confusão - SVM")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()
