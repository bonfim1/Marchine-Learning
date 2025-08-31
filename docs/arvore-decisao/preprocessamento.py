#pré processamento


# Verificar valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum())

# Remover linhas com valores ausentes
df = df.dropna()

# Separar variáveis explicativas (X) e variável alvo (y)
X = df.drop(["quality", "target"], axis=1)  # Todas as colunas menos quality e target
y = df["target"]                             # Apenas a coluna target

# Transformar variáveis categóricas em variáveis dummy (0 ou 1)
X = pd.get_dummies(X, drop_first=True)       # drop_first=True evita multicolinearidade

# Normalizar os dados numéricos
scaler = StandardScaler()                    # Cria o objeto para padronização
X_scaled = scaler.fit_transform(X)           # Calcula média e desvio e transforma os dados


# 3. Divisão dos Dados


# Separar em treino (70%) e teste (30%), mantendo proporção da classe (stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} registros")
print(f"Tamanho teste: {X_test.shape[0]} registros")
# Converte variáveis categóricas em dummies (se tiver).
#pd.get_dummies(X) → cria colunas dummies para todas variáveis categóricas.
#drop_first=True → remove a primeira categoria de cada variável (para evitar multicolinearidade, ou seja, colunas redundantes).
# Divide em treino (70%) e teste (30%).



