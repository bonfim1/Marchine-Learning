# Visualizar a árvore
plt.figure(figsize=(18,10))
plot_tree(model, feature_names=X.columns, class_names=["ruim","bom"], filled=True)
plt.show()