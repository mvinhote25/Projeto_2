# Predição do Número de Estudantes Desvinculados em Cursos de Graduação no Brasil Utilizando Técnicas de Machine Learning

## 1. Sobre o projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de estimar o número de estudantes desvinculados em cursos de graduação no Brasil, utilizando dados do Censo da Educação Superior.

A proposta é investigar em que medida características acadêmicas, institucionais, geográficas e administrativas estão associadas ao número de estudantes desvinculados, além de avaliar a capacidade preditiva de modelos de regressão.

O projeto utiliza Python e bibliotecas de Ciência de Dados e Machine Learning, com desenvolvimento realizado em notebooks Jupyter.

## 2. Objetivo

Desenvolver e avaliar modelos de regressão capazes de estimar a variável-alvo `QT_SIT_DESVINCULADO`, que representa o número de estudantes com situação de vínculo classificada como desvinculada.

O problema é tratado como uma tarefa de **regressão supervisionada**, na qual as características disponíveis nas observações do Censo são utilizadas para estimar o valor da variável-alvo.

A base utilizada contém dados agregados. Portanto, o modelo não realiza previsões individualizadas sobre estudantes, mas estima quantidades de estudantes desvinculados nas observações disponíveis.

## 3. Organização do projeto

O desenvolvimento está organizado em duas etapas principais.

### Parte 1 — Preparação dos dados

Notebook: `Projeto_2_Parte_1(1).ipynb`

Esta etapa contempla:

- Seleção das variáveis relevantes para o problema;
- Análise exploratória dos dados;
- Identificação e tratamento de inconsistências nas variáveis categóricas;
- Investigação e tratamento de valores ausentes;
- Identificação de possíveis valores extremos;
- Análise da assimetria das variáveis numéricas;
- Aplicação de transformações logarítmicas às variáveis de contagem selecionadas;
- Análise de correlação e investigação da multicolinearidade;
- Identificação das variáveis numéricas e categóricas;
- Exportação da base preparada para utilização na etapa de modelagem.

O resultado desta etapa é o arquivo `dados_preparados.csv`, que contém a variável-alvo, os preditores selecionados e as variáveis categóricas ainda não codificadas.

### Parte 2 — Modelagem e avaliação

Notebook: `Projeto_2_Parte_2(3).ipynb`

Nesta etapa são realizadas as seguintes atividades:

- Carregamento e verificação da base preparada;
- Separação da variável-alvo `QT_SIT_DESVINCULADO` e das variáveis preditoras;
- Divisão dos dados em conjuntos de treino e teste;
- Construção de um pipeline de pré-processamento;
- Codificação das variáveis categóricas por meio de One-Hot Encoding;
- Imputação de valores ausentes, quando necessário;
- Padronização das variáveis numéricas;
- Treinamento do modelo ElasticNet sem filtragem da variável-alvo;
- Análise dos erros absolutos;
- Filtragem de observações com valores elevados da variável-alvo;
- Otimização dos hiperparâmetros do ElasticNet por validação cruzada;
- Avaliação do modelo nos conjuntos de treino filtrado, teste filtrado e teste completo;
- Salvamento do modelo final.

## 4. Preparação e pré-processamento

As variáveis foram selecionadas considerando sua relevância para a descrição das características acadêmicas, institucionais e geográficas das observações.

Na etapa de preparação, foram investigados valores ausentes, inconsistências e distribuições assimétricas. Algumas variáveis de contagem apresentaram forte assimetria à direita, motivando a aplicação de transformações logarítmicas.

As variáveis categóricas foram mantidas em sua representação original durante a preparação da base. Na modelagem, elas são codificadas por One-Hot Encoding, enquanto as variáveis numéricas são padronizadas por meio do `StandardScaler`.

Essas operações são incorporadas ao pipeline de modelagem, de modo que o pré-processamento seja ajustado durante o treinamento, evitando que informações do conjunto de teste participem do ajuste das transformações.

## 5. Modelo de Machine Learning

O algoritmo utilizado é o **ElasticNet**, um modelo de regressão linear regularizada que combina as penalizações L1 e L2.

Essa abordagem permite controlar a complexidade do modelo e pode ser útil quando existem associações entre as variáveis preditoras.

Os hiperparâmetros `alpha` e `l1_ratio` são investigados por meio de `GridSearchCV`, utilizando validação cruzada com três divisões e o erro absoluto médio (MAE) como critério de seleção.

A avaliação dos modelos utiliza as seguintes métricas:

- **MAE (Mean Absolute Error):** mede o erro absoluto médio das previsões, expresso na escala da variável-alvo.
- **RMSE (Root Mean Squared Error):** penaliza mais fortemente erros de grande magnitude.
- **R² (coeficiente de determinação):** avalia quanto da variabilidade da variável-alvo é explicada pelas previsões do modelo.

## 6. Experimento com filtragem da variável-alvo

Além do treinamento com todos os valores da variável-alvo, foi conduzido um experimento em que as observações com valores superiores ao limite definido pelo quantil 0,88 são excluídas do conjunto de treino.

O limite é calculado exclusivamente a partir da variável-alvo do conjunto de treino. O mesmo limite é utilizado para identificar o subconjunto correspondente no teste.

Essa estratégia permite investigar se o modelo apresenta melhor desempenho na faixa predominante da distribuição de `QT_SIT_DESVINCULADO`.

A avaliação considera três conjuntos:

1. **Treino filtrado:** observações utilizadas no ajuste do modelo;
2. **Teste filtrado:** observações do teste cujos valores reais estão abaixo ou iguais ao limite estabelecido;
3. **Teste completo:** todas as observações do conjunto de teste, incluindo os valores extremos.

Os resultados indicaram desempenho mais consistente nos conjuntos filtrados. Entretanto, a avaliação no teste completo evidenciou dificuldades na previsão das observações de maior magnitude.

A filtragem não significa que os valores extremos sejam necessariamente erros nos dados. Ela representa uma restrição do escopo do modelo, que deve ser considerada na interpretação dos resultados.

## 7. Modelo salvo

O notebook da etapa de modelagem salva o modelo final otimizado no arquivo:

`modelo_quantil_elasticnet_otimizado.pkl`

O arquivo é gerado utilizando a biblioteca `joblib`.

O modelo final corresponde à versão treinada com os dados filtrados e com os hiperparâmetros selecionados pela validação cruzada. O pipeline inclui o pré-processamento das variáveis numéricas e categóricas.

Para utilizar o modelo posteriormente, é necessário carregá-lo com `joblib` e fornecer novos dados com as mesmas variáveis preditoras utilizadas no treinamento, respeitando seus nomes e tipos.

## 8. Tecnologias utilizadas

- Python;
- Jupyter Notebook;
- Pandas;
- NumPy;
- Matplotlib e Seaborn;
- Scikit-learn;
- Joblib;
- Biblioteca `watermark` para registro de informações do ambiente.

## 9. Como executar o projeto

1. Disponibilize os notebooks e o arquivo `dados_preparados.csv` no ambiente de execução.
2. Confira e ajuste os caminhos de leitura dos arquivos nos notebooks, se necessário.
3. Execute primeiro o notebook da Parte 1 para reproduzir a preparação dos dados, caso deseje reconstruir a base.
4. Execute o notebook da Parte 2 para treinar, otimizar e avaliar os modelos.
5. Após a execução da etapa de modelagem, o arquivo `modelo_quantil_elasticnet_otimizado.pkl` será salvo no diretório de trabalho do notebook.

### Dependências

As principais bibliotecas necessárias podem ser instaladas com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib watermark
```

## 10. Limitações e trabalhos futuros

Os resultados indicam que o ElasticNet consegue estimar a variável-alvo com maior consistência para observações abaixo do limite de filtragem adotado. Entretanto, o desempenho no conjunto de teste completo é inferior, principalmente devido à dificuldade em representar observações com números elevados de estudantes desvinculados.

Entre as possibilidades de continuidade estão:

- Comparar o ElasticNet com outros algoritmos de regressão;
- Investigar modelos mais adequados a variáveis de contagem;
- Analisar os erros por faixas de valores da variável-alvo;
- Avaliar estratégias que reduzam o impacto dos valores extremos sem excluí-los;
- Investigar outras variáveis preditoras e possíveis fontes de informação;
- Avaliar o desempenho em diferentes anos e contextos institucionais.

Também é importante explicitar que a base não permite acompanhar necessariamente o mesmo curso ao longo do tempo por meio de um identificador único adequado. Portanto, o objetivo desta versão é estimar o número de estudantes desvinculados a partir das características disponíveis na mesma observação, e não prever diretamente o valor futuro de cada curso.

## 11. Considerações finais

O projeto apresenta um fluxo de trabalho de Ciência de Dados que abrange preparação dos dados, análise exploratória, pré-processamento, modelagem supervisionada, otimização de hiperparâmetros e avaliação de desempenho.

Os experimentos realizados permitem compreender melhor as limitações de um modelo linear regularizado diante de uma variável-alvo assimétrica e com valores extremos. Os resultados também destacam a importância de alinhar o objetivo de previsão à população para a qual o modelo efetivamente apresenta desempenho satisfatório.

O modelo desenvolvido constitui uma etapa exploratória para investigar o potencial de técnicas de Machine Learning na análise de dados do Censo da Educação Superior e pode servir de base para experimentos posteriores voltados ao apoio à análise e à tomada de decisões no contexto educacional.
