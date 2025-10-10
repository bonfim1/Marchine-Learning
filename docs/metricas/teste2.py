# ============================================================
# Projeto: Classificação e Clustering no Dataset Wine Quality
# ============================================================

# --- IMPORTAÇÕES ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, silhouette_score
)
from imblearn.over_sampling import SMOTE
from warnings import filterwarnings

# --- CONFIGURAÇÕES GERAIS ---
filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_palette("viridis")
np.random.seed(42)

# ============================================================
# ETAPA 1 – CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ============================================================
def carregar_dados():
    """Baixa e carrega o dataset de vinhos."""
    print("\n📥 Baixando dataset...")
    path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality")
    df = pd.read_csv(os.path.join(path, "wine_quality_merged.csv"), index_col=0)
    print("✅ Dataset carregado com sucesso!")
    return df


def explorar_dados(df):
    """Mostra informações básicas e visualizações iniciais."""
    print("\n--- ETAPA 1: EXPLORAÇÃO DOS DADOS ---")
    print("\nColunas disponíveis:", df.columns.tolist())
    print("\nInformações gerais:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())

    # Distribuição da qualidade
    plt.figure(figsize=(8,5))
    sns.countplot(x="quality", data=df)
    plt.title("Distribuição da Qualidade Original")
    plt.show()

    # Correlação
    plt.figure(figsize=(12,8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mapa de Correlação")
    plt.show()


# ============================================================
# ETAPA 2 – PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar_dados(df):
    """Limpeza, criação de variável alvo e codificação."""
    print("\n--- ETAPA 2: PRÉ-PROCESSAMENTO ---")

    df.dropna(inplace=True)

    # Criar variável alvo binária: bom (>=6) ou ruim (<6)
    df["target"] = (df["quality"] >= 6).astype(int)
    print("Distribuição da variável alvo:")
    print(df["target"].value_counts())

    # Separação X e y
    X = df.drop(["quality", "target"], axis=1)
    y = df["target"]

    # One-hot encoding (caso exista coluna de tipo de vinho)
    for col in ["wine_type", "type"]:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
            print(f"Coluna categórica '{col}' convertida em dummies.")
            break

    print("\nColunas finais:", X.columns.tolist())
    return X, y


# ============================================================
# ETAPA 3 – DIVISÃO E NORMALIZAÇÃO DOS DADOS
# ============================================================
def preparar_treino_teste(X, y):
    """Divide os dados, normaliza e aplica balanceamento (SMOTE)."""
    print("\n--- ETAPA 3: DIVISÃO DOS DADOS ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print("Tamanho treino:", X_train.shape, "| teste:", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balanceamento
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print("Distribuição após SMOTE:")
    print(y_train_bal.value_counts())

    return X_train_bal, X_test_scaled, y_train_bal, y_test, scaler


# ============================================================
# ETAPA 4 – TREINAMENTO KNN
# ============================================================
def treinar_knn(X_train, y_train, k=7):
    """Treina o modelo KNN com o valor de k fornecido."""
    print("\n--- ETAPA 4: TREINAMENTO KNN ---")
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"Modelo KNN treinado com k={k}")
    return model


# ============================================================
# ETAPA 5 – AVALIAÇÃO KNN
# ============================================================
def avaliar_knn(model, X_test, y_test):
    """Avalia o modelo e busca o melhor K."""
    print("\n--- ETAPA 5: AVALIAÇÃO KNN ---")

    y_pred = model.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Acurácia Balanceada:", balanced_accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Ruim", "Bom"]).plot(cmap="viridis")
    plt.title("Matriz de Confusão – KNN")
    plt.show()

    # Busca do melhor K
    k_values = range(1, 30, 2)
    f1_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_bal, y_train_bal)
        preds = knn.predict(X_test)
        f1_scores.append(f1_score(y_test, preds, average="weighted"))

    best_k = k_values[np.argmax(f1_scores)]
    print("Melhor k baseado em F1-Score:", best_k)

    plt.figure(figsize=(10,5))
    plt.plot(k_values, f1_scores, marker="o")
    plt.title("F1-Score vs k")
    plt.xlabel("k")
    plt.ylabel("F1-Score")
    plt.axvline(best_k, color="red", linestyle="--")
    plt.show()


# ============================================================
# ETAPA 6 – CLUSTERING K-MEANS
# ============================================================
def aplicar_kmeans(X, df):
    """Executa o clustering e avalia a qualidade dos clusters."""
    print("\n--- ETAPA 6: CLUSTERING K-MEANS ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = clusters

    silhouette = silhouette_score(X_scaled, clusters)
    print("Silhouette Score:", silhouette)
    print("Centroides:\n", kmeans.cluster_centers_)

    plt.figure(figsize=(8,5))
    sns.countplot(x="kmeans_cluster", data=df)
    plt.title("Distribuição dos Clusters K-Means")
    plt.show()


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    df = carregar_dados()
    explorar_dados(df)
    X, y = preprocessar_dados(df)
    X_train_bal, X_test_scaled, y_train_bal, y_test, scaler = preparar_treino_teste(X, y)
    knn_model = treinar_knn(X_train_bal, y_train_bal)
    avaliar_knn(knn_model, X_test_scaled, y_test)
    aplicar_kmeans(X, df)
