# Importância das variáveis
importances = model.feature_importances_                # Extrai importância de cada feature
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Gráfico das variáveis mais importantes
sns.barplot(x=feat_imp, y=feat_imp.index, color=cor_vinho)
plt.title("Importância das Variáveis na Árvore")
plt.show()