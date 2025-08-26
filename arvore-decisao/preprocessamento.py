
# Pré-processamento

print("\nValores ausentes:")
print(df.isnull().sum())

# Features: tirando quality e target
X = df.drop(["quality", "target"], axis=1)
y = df["target"]

# Se houver variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")

# Converte variáveis categóricas em dummies (se tiver).
#pd.get_dummies(X) → cria colunas dummies para todas variáveis categóricas.
#drop_first=True → remove a primeira categoria de cada variável (para evitar multicolinearidade, ou seja, colunas redundantes).
# Divide em treino (70%) e teste (30%).