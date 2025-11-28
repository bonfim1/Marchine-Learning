# ============================================================
# ===================== IMPORTAÇÕES ==========================
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import networkx as nx

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# ===================== PALETA VINHO =========================
# ============================================================

paleta_vinho = ["#7B1113", "#A02C2D", "#C44B4D", "#D67B7C", "#F2A6A6"]


# ============================================================
# ========== ETAPA 1: EXPLORAÇÃO DOS DADOS ===================
# ============================================================

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
plt.title("Distribuição da Qualidade por Tipo de Vinho", fontsize=13, weight="bold")
plt.show()


# ============================================================
# ================ ETAPA 2: PRÉ-PROCESSAMENTO =================
# ============================================================

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
# =============== ETAPA 3: DIVISÃO DOS DADOS =================
# ============================================================

print("\n--- ETAPA 3: DIVISÃO DOS DADOS ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


# ============================================================
# ================= ETAPA 4: TREINAMENTO =====================
# ============================================================

print("\n--- ETAPA 4: TREINAMENTO ---")

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42
)
modelo.fit(X_train, y_train)


# ============================================================
# ================= ETAPA 5: AVALIAÇÃO =======================
# ============================================================

print("\n--- ETAPA 5: AVALIAÇÃO ---")

y_pred = modelo.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.3f}")

# MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Reds", fmt='d', cbar=False,
            xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# IMPORTÂNCIA DAS VARIÁVEIS
importances = pd.Series(modelo.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
importances.sort_values().tail(10).plot(kind="barh", color="#7B1113")
plt.title("Top 10 - Importância das Variáveis")
plt.show()


# ============================================================
# =================== ETAPA 6: PAGE RANK =====================
# ============================================================

print("\n--- ETAPA 6: PAGE RANK ---")

# Selecionar variáveis numéricas
df_num = df.select_dtypes(include=np.number)

# Matriz de correlação
corr = df_num.corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr, cmap=paleta_vinho[::-1], linewidths=0.5)
plt.title("Correlação entre Variáveis — Paleta Vinho", fontsize=14, weight="bold")
plt.show()

# Criar grafo baseado em correlação
limiar = 0.35
G = nx.DiGraph()

for i in corr.index:
    for j in corr.columns:
        if i != j:
            peso = corr.loc[i, j]
            if abs(peso) >= limiar:
                G.add_edge(i, j, weight=abs(peso))

print(f"Grafo criado com {len(G.nodes())} nós e {len(G.edges())} arestas.")

# PageRank
pr = nx.pagerank(G, alpha=0.85, weight="weight")

pagerank_df = (
    pd.DataFrame(pr.items(), columns=["variavel", "pagerank"])
    .sort_values("pagerank", ascending=False)
)

print("\nRanking das variáveis por PageRank:")
print(pagerank_df)

# GRÁFICO 1 — Grafo com design vinho
plt.figure(figsize=(14,11))
pos = nx.spring_layout(G, seed=42)
node_sizes = [v * 3500 + 200 for v in pr.values()]
node_colors = sns.color_palette(paleta_vinho, n_colors=len(G.nodes()))
edge_weights = [d['weight'] * 3 for (_,_,d) in G.edges(data=True)]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       edgecolors="#300000", linewidths=1.5, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color="#7B1113")
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Grafo das Variáveis — PageRank (Paleta Vinho)", fontsize=16, weight="bold")
plt.axis("off")
plt.show()

# GRÁFICO 2 — Barras PageRank
plt.figure(figsize=(10,6))
sns.barplot(data=pagerank_df, x="pagerank", y="variavel", palette=paleta_vinho)
plt.title("Ranking de Importância (PageRank)", fontsize=15, weight="bold")
plt.xlabel("Score de PageRank")
plt.ylabel("Variável")
plt.show()

