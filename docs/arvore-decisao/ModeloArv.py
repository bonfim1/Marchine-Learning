
# Modelo Árvore de Decisão

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Avaliação
y_pred = model.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
