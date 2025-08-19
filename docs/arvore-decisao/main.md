## Exploração de dados
A variável target foi criada a partir da coluna quality, classificando os vinhos em duas categorias:

0 (ruim): vinhos com qualidade menor que 6;

1 (bom): vinhos com qualidade maior ou igual a 6.

No gráfico abaixo, é possível observar um forte desbalanceamento entre as classes. A grande maioria dos vinhos foi classificada como “bons” (1), enquanto apenas uma pequena parcela foi classificada como “ruins” (0).

Resumo do conjunto de dados
Total de linhas: 6.497
Total de colunas: 12
Variável de destino: quality (valores inteiros de 3 a 9)
Formato de arquivo: CSV (UTF-8)
Valores ausentes: Nenhum
Duplicatas removidas: Sim

## Exploração dos Dados	Gráfico
![GráficoRB](image/GráficoBR.png)

## Exploração dos Dados	Código
``` python
--8<-- "./docs/arvore-decisao/teste.py"
```



