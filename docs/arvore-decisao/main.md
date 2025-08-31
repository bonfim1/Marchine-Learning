## Exploração de dados
A variável target foi criada a partir da coluna quality, classificando os vinhos em duas categorias:

0 (ruim): vinhos com qualidade menor que 5;

1 (bom): vinhos com qualidade maior ou igual a 5.

No gráfico abaixo, é possível observar um forte desbalanceamento entre as classes. A grande maioria dos vinhos foi classificada como “bons” (1), enquanto apenas uma pequena parcela foi classificada como “ruins” (0).

Resumo do conjunto de dados
Total de linhas: 6.497
Total de colunas: 12
Variável de destino: quality (valores inteiros de 3 a 9)
Formato de arquivo: CSV (UTF-8)
Valores ausentes: Nenhum
Duplicatas removidas: Sim

1   -  6251

0   -  246


=== "Exploração dos Dados	Gráfico"
    ![GráficoRB](image/GráficoRb.png)

=== "Exploração dos Dados	Código"

``` python
--8<-- "./docs/arvore-decisao/teste.py"
```
### Distribuição da Qualidade do Vinho

=== "Gráfico de Distribuição da Qualidade do Vinho"
    ![GráficoRB](image/Distri.png)

=== "Código de Distribuição da Qualidade do Vinho"
``` python
--8<-- "./docs/arvore-decisao/DistribuVinho.py"
```


É um histograma com curva KDE que mostra como as notas de qualidade se distribuem.

Exemplo do gráfico:

A maioria dos vinhos está entre 5 e 6 pontos.

Poucos vinhos têm nota 3 ou 8+.

A distribuição mostra que a maior parte dos vinhos está concentrada em notas médias (5-6). Isso ajuda a entender a base de dados e a justificar a criação da variável target binária (bom >=5, ruim <5).


###  Correlação entre Variáveis Numéricas

=== "Mapa de Correlação entre Variáveis Numéricas"
    ![GráficoRB](image/Mapa de Correlação numerica.png)

=== " Código Mapa de Correlação entre Variáveis Numéricas"
``` python
--8<-- "./docs/arvore-decisao/correlação.py"
```

heatmap de correlação mostra como cada variável se relaciona com as outras.

Os valores variam de -1 a 1:

1: correlação positiva perfeita (quando uma aumenta, a outra também aumenta)

-1: correlação negativa perfeita (uma aumenta e a outra diminui)

0: nenhuma correlação linear

Cores mais escuras indicam maior correlação positiva.

O mapa de correlação permite identificar relações lineares entre variáveis. Correlações altas podem indicar redundância de informação, enquanto correlações baixas indicam independência entre atributos.

## Pré-processamento
1-Verifica se há valores nulos.

2-Define features (X) e variável alvo (y).

3-Converte variáveis categóricas em dummies (se tiver).

4-Divide em treino (70%) e teste (30%).

Tamanho treino: 4547 registros

Tamanho teste: 1950 registros

=== "Código Pré-processamento"
``` python
--8<-- "./docs/arvore-decisao/preprocessamento.py"
```
## Explicação do Pré-processamento
Verificou-se a ausência de valores faltantes no dataset. As variáveis independentes foram selecionadas, excluindo-se quality e target. As variáveis categóricas foram codificadas por meio de dummies (one-hot encoding) para permitir a leitura pelo modelo. Por fim, o conjunto de dados foi dividido em treino (70%) e teste (30%), de forma estratificada para manter a proporção da variável alvo

Foi investigada a presença de valores zero nas colunas do dataset. A análise revelou que as colunas citric acid e a variável alvo target continham valores iguais a zero.

•
Coluna citric acid: Um valor zero nesta coluna representa a ausência de ácido cítrico no vinho, o que é uma medição química válida e relevante para a análise da qualidade do vinho. Portanto, esses valores foram mantidos.

•
Coluna target: A variável target foi criada para classificar os vinhos em 'ruim' (0) e 'bom' (1). O valor zero é uma das duas classes do nosso modelo de classificação e, portanto, é essencial para o treinamento e a avaliação do mesmo.
 
Ao remove valores ausentes e separa as features (X) da variável alvo (y). Colunas categóricas são transformadas em dummies binárias e todos os dados numéricos são normalizados com StandardScaler para manter a mesma escala, garantindo que o modelo de árvore de decisão aprenda de forma eficiente.

=== "Modelo Árvore de Decisão"
``` python
--8<-- "./docs/arvore-decisao/ModeloArv.py"
```
## Descrição

Treinamento do modelo- Cria uma árvore de decisão limitada a profundidade 5 (para não crescer demais) e treina com os dados de treino.

Avaliação- az previsões com os dados de teste e calcula métricas

O modelo foi avaliado nos dados de teste com métricas de desempenho. A acurácia foi de cerca de 96%, mostrando boa performance geral. A precisão, recall e F1-score foram altos para vinhos bons, mas baixos para vinhos ruins, devido ao desbalanceamento do conjunto de dados.

Acurácia: mede a proporção total de acertos do modelo. No experimento, o modelo apresentou uma acurácia de aproximadamente 96%, indicando bom desempenho geral.

Precisão: avalia, dentre as amostras previstas como “boas”, quantas realmente pertenciam a essa classe. O modelo mostrou alta precisão para a classe majoritária (vinhos bons).

Recall: mede a capacidade de identificar corretamente todos os casos da classe positiva. O recall também foi alto para vinhos bons, mas baixo para vinhos ruins, devido ao desbalanceamento da base.

F1-Score: combina precisão e recall em uma única métrica. Os valores foram satisfatórios para vinhos bons, mas menores para vinhos ruins, reforçando a dificuldade do modelo em prever a classe minoritária.


=== "Imagem Visualizar da Matriz"
    ![Matriz](image/Matriz.png)
    
=== "Código da visualização da Matriz"

``` python
--8<-- "./docs/arvore-decisao/Matriz.py"
```

## Descrição Matriz

A matriz de confusão mostra os seguintes resultados:

Verdadeiro Negativo (TN) = 5 → vinhos ruins corretamente classificados como ruins.

Falso Positivo (FP) = 69 → vinhos ruins incorretamente classificados como bons.

Falso Negativo (FN) = 6 → vinhos bons incorretamente classificados como ruins.

Verdadeiro Positivo (TP) = 1870 → vinhos bons corretamente classificados como bons.

Isso indica que o modelo tem ótimo desempenho na identificação de vinhos bons (classe 1), conseguindo classificar corretamente a grande maioria desses casos. No entanto, o desempenho é muito baixo para vinhos ruins (classe 0), já que praticamente não consegue reconhecê-los.

Esse comportamento está diretamente relacionado ao desbalanceamento do conjunto de dados, onde os vinhos bons são muito mais numerosos que os ruins. Como consequência, o modelo "aprende" a priorizar a classe majoritária (bons) e tem dificuldade em identificar a classe minoritária (ruins).

### Importância das Variáveis na Árvore

=== "Imagem Visualizar a árvore"
    ![arvore](image/Importancia da variavel na Arv.png)
                                                                                                                                                            
=== "Código da visualização da árvore"

        ``` python
        --8<-- "./docs/arvore-decisao/importancia.py"
        ```

O gráfico mostra quais atributos mais influenciam a decisão do modelo.

Cada barra representa uma variável do dataset; quanto maior a barra, mais importante ela é para classificar um vinho como bom ou ruim.

Exemplo do gráfico:

free sulfur dioxide e volatile acidity são as mais importantes.

Variáveis como residual sugar e density têm pouca ou nenhuma importância para a árvore.

O modelo identifica que a quantidade de dióxido de enxofre livre e a acidez volátil são os principais fatores que determinam se um vinho é classificado como bom ou ruim. Outras variáveis têm menor influência.


### Árvore de Decisão

=== "Imagem Visualizar a árvore"
    ![arvore](image/arvore.png)
                                                                                                                                                            
=== "Código da visualização da árvore"

        ``` python
        --8<-- "./docs/arvore-decisao/arvore.py"
        ```
## Estrutura da Árvore

•
Nó Raiz: O nó superior da árvore é o nó raiz, que inicia o processo de tomada de decisão. Neste caso, a primeira divisão é baseada na característica volatile acidity.

•
Nós Internos: Cada nó interno (não folha) contém uma condição de teste (por exemplo, volatile acidity <= 0.39). Se a condição for verdadeira, o caminho da esquerda é seguido; caso contrário, o caminho da direita é seguido.

•
Nós Folha: Os nós folha são os nós terminais da árvore e contêm a previsão da classe (class = bom ou class = ruim). A cor dos nós (azul para 'bom' e laranja para 'ruim') indica a classe majoritária naquele nó.

## Interpretação dos Nós

Cada nó na árvore exibe as seguintes informações:

•
Condição de Divisão: A característica e o valor de corte usados para dividir os dados (ex: volatile acidity <= 0.39).

•
gini: O coeficiente de Gini, que mede a impureza do nó. Um valor de Gini próximo de 0 indica um nó mais puro (ou seja, a maioria das amostras pertence a uma única classe).

•
samples: O número de amostras de treinamento que atingiram aquele nó.

•
value: A contagem de amostras para cada classe ([ruim, bom]) naquele nó. Por exemplo, 
value = [1750, 4747] significa que o nó contém 1750 amostras da classe 'ruim' e 4747 amostras 
da classe 'bom'.

•
class: A classe majoritária no nó, que seria a previsão se a decisão terminasse ali.



## Conclusão final - Relatório de Classificação 

Accuracy (Acurácia): 96% → parece muito bom, mas é enganador.

Precision e Recall para classe 0 (ruim):

Precision = 0.45 → Quando ele diz que é ruim, só 45% estão realmente corretos.

Recall = 0.07 → Ele só encontra 7% dos vinhos ruins!

Classe 1 (bom): quase perfeito, acerta praticamente todos.

 Conclusão: o modelo é ótimo para prever vinhos bons, mas péssimo para prever vinhos ruins.