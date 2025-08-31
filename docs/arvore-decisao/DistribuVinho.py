# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())                       # Média, desvio padrão, min, max, quartis

# Visualização da distribuição da qualidade
sns.histplot(df["quality"], bins=7, kde=True, color=cor_vinho)
plt.title("Distribuição da Qualidade do Vinho")
plt.show()
