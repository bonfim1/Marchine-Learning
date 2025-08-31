# Mapa de correlação entre colunas numéricas
numeric_df = df.select_dtypes(include=np.number)   # Seleciona apenas colunas numéricas
sns.heatmap(numeric_df.corr(), cmap="Reds", annot=False)
plt.title("Mapa de Correlação entre Variáveis Numéricas")
plt.show()