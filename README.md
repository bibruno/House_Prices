# Análise de Preços de Venda de Casas

Este projeto tem como objetivo prever os preços de venda de casas com base em diversas características. Para alcançar esse objetivo, foram desenvolvidos quatro códigos principais que se complementam e desempenham papéis distintos no fluxo de trabalho. Abaixo, explicamos a funcionalidade de cada um.

## Fone dos dados: Kaggle.
Link para o data set:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## 1. Pré-processamento de Dados

O primeiro código é responsável pelo pré-processamento dos dados. Ele carrega o conjunto de dados, trata valores ausentes e transforma variáveis categóricas em colunas numéricas usando a técnica de *one-hot encoding*. Além disso, cria novas variáveis que capturam informações relevantes, como a soma das áreas totais da casa e a quantidade total de banheiros. Este passo é crucial para garantir que o modelo receba dados limpos e estruturados, facilitando a análise subsequente.

## 2. Modelagem e Treinamento
![Histogram](https://i.imgur.com/eSQ70Sh.gif)
O segundo código foca na modelagem preditiva. Após carregar e pré-processar os dados, ele divide o conjunto em dados de treino e teste, escalona as variáveis e treina um modelo de regressão XGBoost. A modelagem é uma etapa vital, pois permite a geração de previsões a partir dos dados tratados. O uso de *GridSearchCV* para ajustar hiperparâmetros garante que o modelo esteja otimizado para obter o melhor desempenho possível.



## 3. Avaliação do Modelo
![Resultado](https://i.imgur.com/KRAkete.png)
O terceiro código avalia a performance do modelo treinado. Ele calcula métricas como o Erro Quadrático Médio (MSE) e o Coeficiente de Determinação (R²), que ajudam a entender a precisão das previsões em relação aos preços de venda reais. Além disso, gera gráficos que ilustram a relação entre preços reais e previstos, bem como a distribuição dos erros de previsão. Esta avaliação é fundamental para verificar a eficácia do modelo e identificar áreas para melhorias.

![Erros](https://i.imgur.com/bzFN84c.png)


## 4. Análise Exploratória dos Dados
![Iteração](https://i.imgur.com/JRkjWRx.gif)
O quarto código realiza uma análise exploratória dos dados. Ele investiga as interações entre variáveis relevantes e o preço de venda, como a área total da casa, a qualidade geral e o número de garagens. Através de análises estatísticas e visualizações gráficas, essa etapa fornece insights valiosos sobre as relações nos dados, ajudando a informar decisões sobre quais características são mais influentes no preço de venda. Essa análise não só enriquece a compreensão dos dados, mas também pode guiar ajustes no modelo preditivo.

![Importância](https://i.imgur.com/FfH5RNK.png)

## Conclusão

Juntos, esses códigos formam um fluxo de trabalho robusto para a previsão de preços de venda de casas. Desde o pré-processamento e modelagem até a avaliação e análise exploratória, cada componente desempenha um papel crucial na geração de insights acionáveis e na melhoria da precisão das previsões.
