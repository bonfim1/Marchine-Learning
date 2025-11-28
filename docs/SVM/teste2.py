# ===========================================
# IMPORTS
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import seaborn as sns
import os
import kagglehub
from imblearn.over_sampling import SMOTE

# Configuração Estética
cor_vinho = "#722f37"
sns.set(style="whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[cor_vinho])


# ===========================================
# CARREGAR DATASET
# ===========================================
print("Baixando dataset...")
path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
file_path = os.path.join(path, "wine_quality_merged.csv")
df = pd.read_csv(file_path, index_col=0)
print("Dataset carregado!\n")

print(df.head())


# ===========================================
# 1. EXPLORAÇÃO DOS DADOS
# ===========================================
print(df.info())
print(df.describe())

# Distribuição da qualidade
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="quality", color=cor_vinho)
plt.title("Distribuição das notas de qualidade", color=cor_vinho)
plt.xlabel("Qualidade")
plt.ylabel("Frequência")
plt.show()

# Pairplot
sns.pairplot(df.iloc[:, :5], plot_kws={'color': cor_vinho})
plt.show()


# ===========================================
# 2. PRÉ-PROCESSAMENTO
# ===========================================
def classificar(q):
    if q <= 5:
        return "ruim"
    elif q == 6:
        return "medio"
    else:
        return "bom"

df["label"] = df["quality"].apply(classificar)

df = pd.get_dummies(df, columns=["type"], drop_first=True)

X = df.drop(columns=["quality", "label"])
y = df["label"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===========================================
# 3. APLICAR SMOTE – BALANCEAMENTO
# ===========================================
print("\nBalanceando base com SMOTE...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Antes do SMOTE:")
print(y.value_counts())
print("\nDepois do SMOTE:")
print(y_resampled.value_counts())

# GRÁFICO 1 – Distribuição antes e depois do SMOTE
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x=y, color=cor_vinho)
plt.title("Antes do SMOTE", color=cor_vinho)

plt.subplot(1,2,2)
sns.countplot(x=y_resampled, color=cor_vinho)
plt.title("Depois do SMOTE", color=cor_vinho)

plt.tight_layout()
plt.show()


# ===========================================
# 4. DIVISÃO TREINO/TESTE (APÓS SMOTE)
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled
)


# ===========================================
# 5. TREINAMENTO DO MODELO SVM (RBF)
# ===========================================
model = SVC(kernel="rbf", C=2, gamma="scale")
model.fit(X_train, y_train)


# ===========================================
# 6. AVALIAÇÃO DO MODELO
# ===========================================
y_pred = model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(cor_vinho, as_cmap=True))
plt.title("Matriz de Confusão - SVM", color=cor_vinho)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()


# ===========================================
# 7. GRÁFICO 2 – Importância dos atributos via SVM Linear
# ===========================================
print("\nGerando gráfico de importância dos atributos...")

svm_linear = LinearSVC(max_iter=5000)
svm_linear.fit(X_resampled, y_resampled)

importancias = np.mean(np.abs(svm_linear.coef_), axis=0)

plt.figure(figsize=(10,6))
plt.barh(X.columns, importancias, color=cor_vinho)
plt.title("Importância dos Atributos (SVM Linear)", color=cor_vinho)
plt.xlabel("Peso Médio Absoluto")
plt.tight_layout()
plt.show()
