# Análise de Preços de Venda de Casas

Este projeto tem como objetivo prever os preços de venda de casas com base em diversas características. Para alcançar esse objetivo, foram desenvolvidos quatro códigos principais que se complementam e desempenham papéis distintos no fluxo de trabalho. Abaixo, explicamos a funcionalidade de cada um.

## Contexto e Fonte dos Dados

Os dados foram obtidos do Kaggle, um dos maiores portais de competições de ciência de dados e análise preditiva. O dataset utilizado pode ser encontrado [aqui](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

## 1. Pré-processamento de Dados

O primeiro código é responsável pelo pré-processamento dos dados. Ele carrega o conjunto de dados a partir do arquivo train.csv, trata valores ausentes e transforma variáveis categóricas em colunas numéricas usando a técnica de one-hot encoding. Além disso, cria novas variáveis que capturam informações relevantes, como a soma das áreas totais da casa e a quantidade total de banheiros. Este passo é crucial para garantir que o modelo receba dados limpos e estruturados, facilitando a análise subsequente.

## 2. Modelagem

Esse código utiliza dados tratados de um arquivo CSV (train_data_cleaned.csv) para realizar as seguintes análises:

- Gera histogramas para visualizar a distribuição das colunas numéricas.
- Calcula estatísticas descritivas das colunas numéricas.
- Calcula a matriz de correlação entre as colunas numéricas.
- Identifica as correlações das variáveis com a variável alvo (SalePrice).

## 3. Treinamento

Esse código realiza a preparação, treinamento e avaliação de um modelo de regressão XGBoost para prever preços de venda de casas. A limpeza dos dados é feita primeiro, adicionando novas features que combinam outras colunas (como o tamanho total do imóvel ou o número total de banheiros). Valores extremos de preço de venda são removidos usando uma técnica chamada Interquartile Range (IQR) para identificar outliers.

As variáveis independentes (as features) são padronizadas usando StandardScaler, garantindo que todas tenham uma média de 0 e um desvio padrão de 1, o que é importante para o desempenho de muitos modelos de machine learning.

O modelo XGBoost é treinado com uma busca em grade (GridSearchCV) para otimizar os hiperparâmetros, como o número de estimadores e a profundidade máxima das árvores. O melhor modelo encontrado é usado para prever os preços de venda no conjunto de teste. As métricas de desempenho como Mean Squared Error (MSE) e R² são calculadas para avaliar a precisão das previsões.

Além disso, o código gera gráficos para visualizar os resultados:

![Gráfico de Dispersão](https://i.imgur.com/KRAkete.png)
*Gráfico de Dispersão: Preços Previstos vs. Reais*

![Histograma de Erros](https://i.imgur.com/bzFN84c.png)
*Histograma de Erros de Previsão*

Ele também gera uma lista de importâncias das features para identificar quais variáveis mais influenciaram o modelo.

## 4. Análise Exploratória dos Dados

Esse código realiza as seguintes análises:

- Calcula a soma das áreas de porão, primeiro e segundo andares para criar a variável TotalSF.
- Realiza uma análise detalhada das interações entre algumas variáveis e o preço de venda (SalePrice), incluindo TotalSF, OverallQual (Qualidade Geral) e GarageCars (Número de Garagens).
- Gera gráficos para visualizar essas interações:

![Gráfico de Interação](https://i.imgur.com/JRkjWRx.gif)
*Gráfico de Interação: TotalSF vs. SalePrice*

![Importância das Features](https://i.imgur.com/FfH5RNK.png)
*Importância das Features no Modelo*

Essas análises ajudam a entender como as variáveis selecionadas influenciam o preço de venda das casas no conjunto de dados.

## 5. Teste do Modelo

O arquivo `test.py` é responsável por executar o modelo treinado em novos dados. Ele carrega o conjunto de dados de teste, aplica o mesmo pré-processamento que foi utilizado no conjunto de treinamento e realiza previsões de preços de venda utilizando o modelo XGBoost otimizado. Isso permite avaliar o desempenho do modelo em dados não vistos e verificar a eficácia das previsões.

## Conclusão

Juntos, esses códigos formam um fluxo de trabalho robusto para a previsão de preços de venda de casas. Desde o pré-processamento e modelagem até a avaliação e análise exploratória, cada componente desempenha um papel crucial na geração de insights acionáveis e na melhoria da precisão das previsões.

Para um detalhamento técnico mais profundo, consulte o código fonte e os recursos adicionais mencionados ao longo do resumo.
