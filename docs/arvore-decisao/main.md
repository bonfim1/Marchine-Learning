## Exploração de dados

Descrição do dataset

O conjunto de dados contém informações do histórico de músicas tocadas no Spotify, incluindo:

spotify_track_uri: identificador da música
ts: timestamp da reprodução
platform: plataforma usada
ms_played: tempo em milissegundos que a música foi tocada
track_name, artist_name, album_name
reason_start, reason_end, shuffle, skipped

O objetivo do projeto é prever se uma música será pulada ou não (skipped).

-Distribuição da variável alvo skipped

.False (não puladas): ~14.000

.True (puladas): ~20.000

Interpretação:
Há mais músicas puladas do que não puladas, indicando uma leve predominância da classe True.
Embora a distribuição não seja perfeitamente balanceada, há dados suficientes em ambas as classes para treinar o modelo de árvore de decisão.

A variável alvo skipped apresenta 14.000 registros False (não puladas) e 20.000 registros True (puladas). Isso indica que há uma leve predominância de músicas puladas, mas ambas as classes têm quantidade suficiente de dados para treinamento do modelo. O gráfico abaixo ilustra essa distribuição.”

## Exploração dos Dados	
``` python
--8<-- "./docs/arvore-decisao/adecision-tree.py"
```

