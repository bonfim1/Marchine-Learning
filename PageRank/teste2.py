import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns # Adicionado para melhor visualização (heatmap)
from sklearn.preprocessing import StandardScaler
import kagglehub
import requests.exceptions # Importar exceções de requests para tratar o timeout

# Definir paleta vinho
paleta_vinho = ["#7B1113", "#A02C2D", "#C44B4D", "#D67B7C", "#F2A6A6"]
FEATURE = 'volatile acidity'
THRESHOLD_SIMILARIDADE = 0.05 
N_TOP = 10 # Número de nós mais importantes para análise

# ============================================================
# === ETAPA 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ===
print("--- ETAPA 1: PREPARAÇÃO DOS DADOS ---")

# 1. Carregar o Dataset com tratamento de timeout
print("Baixando dataset...")
try:
    # Aumentar o timeout de 5s para 30s
    path = kagglehub.dataset_download("saeedomranpour/red-and-white-wine-quality", timeout=30)
    file_path = os.path.join(path, "wine_quality_merged.csv")
    df = pd.read_csv(file_path, index_col=0)
    print("Dataset de Vinho carregado com sucesso!\n")
except requests.exceptions.ReadTimeout:
    print("ERRO: O download excedeu o tempo limite (30s). Tente novamente mais tarde ou use a Solução 2 (download manual).")
    exit()
except Exception as e:
    print(f"ERRO FATAL ao carregar o dataset do Kaggle: {e}")
    exit()

# Adicionar ID para os nós (index original)
df['node_id'] = df.index
N = len(df)

# 2. Normalizar a feature
scaler = StandardScaler()
df[FEATURE + '_scaled'] = scaler.fit_transform(df[[FEATURE]])

# ============================================================
# === ETAPA 2: MODELAGEM DO GRAFO ===
print("\n--- ETAPA 2: MODELAGEM DO GRAFO ---")

G = nx.DiGraph()
G.add_nodes_from(df['node_id'])

# Criar arestas de similaridade na feature escolhida e qualidade
print("Construindo grafo de similaridade...")
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        
        # Calcular a diferença (similaridade) na feature normalizada
        diff = abs(df.loc[i, FEATURE + '_scaled'] - df.loc[j, FEATURE + '_scaled'])
        
        # Regra de Aresta: Aresta i -> j se forem similares E i tiver qualidade >= j
        if diff <= THRESHOLD_SIMILARIDADE and df.loc[i, 'quality'] >= df.loc[j, 'quality']:
            G.add_edge(i, j)

print(f"Grafo construído com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

# Se o grafo não tiver arestas, saia para evitar erros no PageRank
if G.number_of_edges() == 0:
    print("AVISO: O grafo não tem arestas (Verifique THRESHOLD_SIMILARIDADE). Terminando execução.")
    exit()

# ============================================================
# === ETAPA 3: IMPLEMENTAÇÃO DO PAGERANK (DO ZERO) ===
print("\n--- ETAPA 3: PAGERANK ITERATIVO ---")

def pagerank_custom(G, d=0.85, max_iter=100, tol=1.0e-4):
    """Implementação do PageRank a partir da fórmula iterativa."""
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    pr = {node: 1 / N for node in nodes} # 1. Inicialização
    out_degrees = {node: G.out_degree(node) for node in nodes}
    
    for i in range(max_iter):
        new_pr = {}
        err = 0 
        
        for p_i in nodes:
            rank_sum = 0
            # M(p_i) são os nós que apontam para p_i
            for p_j in G.predecessors(p_i):
                L_j = out_degrees[p_j]
                if L_j > 0:
                    rank_sum += pr[p_j] / L_j
            
            # Aplicar a fórmula: PR(p_i) = (1-d)/N + d * sum(...)
            pr_new = (1 - d) / N + d * rank_sum
            err += abs(pr_new - pr[p_i])
            new_pr[p_i] = pr_new
        
        pr = new_pr
        
        # 4. Critério de Convergência
        if err < tol:
            print(f"Convergiu na iteração {i + 1} para d={d}.")
            break
            
    if i == max_iter - 1:
        print(f"Atingiu o máximo de iterações ({max_iter}) para d={d}.")
        
    return pr

# 5. Calcular o PageRank com d=0.85 (Padrão)
d_default = 0.85
pr_scores_custom = pagerank_custom(G, d=d_default)

# 5. Comparar com implementação pronta (networkx)
pr_scores_nx = nx.pagerank(G, alpha=d_default, tol=1.0e-4)

# Converter para DataFrame para fácil análise
df_pr = pd.DataFrame(
    {'PR_Custom': pr_scores_custom, 'PR_NetworkX': pr_scores_nx}
)
df_pr = df_pr.sort_values(by='PR_Custom', ascending=False)

# Adicionar os dados originais de volta
df_pr = df_pr.merge(df, left_index=True, right_on='node_id')

# ============================================================
# === ETAPA 4: ANÁLISE DOS RESULTADOS ===
print("\n--- ETAPA 4: ANÁLISE DOS RESULTADOS ---")

# Comparação e verificação da implementação
custom_vs_nx = np.allclose(
    df_pr['PR_Custom'].values, df_pr['PR_NetworkX'].values, atol=1e-4
)
print(f"Resultado Customizado vs. NetworkX (d={d_default}) são iguais (tolerância 1e-4): **{custom_vs_nx}**")

# 6. Top N_TOP Nós Mais Importantes
top_n = df_pr.head(N_TOP)
print(f"\nTop {N_TOP} Vinhos (Nós) com Maior PageRank (d={d_default}):")
print(top_n[['node_id', 'type', 'quality', FEATURE, 'PR_Custom']])

# Interpretação
print("\nInterpretação (O que o Alto PageRank representa):")
print(f"No grafo de similaridade de vinhos, os nós (vinhos) com PageRank alto são considerados os **vinhos mais 'autoritários'** na rede de qualidade.")
print("Eles são vinhos que:")
print(f"* **Recebem muitas 'ligações de qualidade'**: São similares em '{FEATURE}' a muitos outros vinhos, especialmente àqueles com qualidade **mais baixa**.")
print("* **Têm alta qualidade (ou similaridade a vinhos de alta qualidade)**: A regra de ligação (i → j, se `quality_i >= quality_j`) faz com que vinhos de melhor qualidade 'votem' em seus pares ou em vinhos de qualidade inferior/igual, mas que são quimicamente semelhantes. Assim, o PR valoriza os vinhos que definem o **padrão de qualidade** para um certo perfil químico.")

# ============================================================
# === ETAPA 5: VARIAÇÃO DO FATOR DE AMORTECIMENTO (d) ===
print("\n--- ETAPA 5: VARIAÇÃO DO FATOR DE AMORTECIMENTO (d) ---")
d_values = [0.5, 0.85, 0.99]
results_d = {}

for d in d_values:
    pr_d = pagerank_custom(G, d=d, tol=1e-5)
    results_d[d] = pd.Series(pr_d)

df_d = pd.DataFrame(results_d)
df_d = df_d.merge(df, left_index=True, right_on='node_id')

# Visualização da variação do ranking
df_plot = df_d.sort_values(by=0.85, ascending=False).head(N_TOP)
df_plot.index = df_plot['node_id']

plt.figure(figsize=(10, 6))
df_plot[[0.5, 0.85, 0.99]].plot(
    kind='bar', color=paleta_vinho[::2]
)
plt.title(f"PageRank dos Top {N_TOP} Vinhos com Diferentes Fatores d")
plt.xlabel("ID do Vinho")
plt.ylabel("PageRank Score")
plt.legend(title='Fator d')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.5)
plt.show() 

