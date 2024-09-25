# Análise de Preços de Venda de Casas

Este projeto tem como objetivo prever os preços de venda de casas com base em diversas características. Para alcançar esse objetivo, foram desenvolvidos quatro códigos principais que se complementam e desempenham papéis distintos no fluxo de trabalho. Abaixo, explicamos a funcionalidade de cada um.

## Fonte dos dados: Kaggle.
Link para o data set:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

## 1. Pré-processamento de Dados

O primeiro código é responsável pelo pré-processamento dos dados. Ele carrega o conjunto de dados, trata valores ausentes e transforma variáveis categóricas em colunas numéricas usando a técnica de *one-hot encoding*. Além disso, cria novas variáveis que capturam informações relevantes, como a soma das áreas totais da casa e a quantidade total de banheiros. Este passo é crucial para garantir que o modelo receba dados limpos e estruturados, facilitando a análise subsequente.

## 2. Modelagem 
![Histogram](https://i.imgur.com/eSQ70Sh.gif)
Aqui temos as correlações das colunas com SalePrice que é a variável a ser alcançada com o modelo preditivo.


## 3. Treinamento
![Resultado](https://i.imgur.com/KRAkete.png)
O treinamento é feito nessa etapa.

![Erros](https://i.imgur.com/bzFN84c.png)


## 4. Análise Exploratória dos Dados
![Iteração](https://i.imgur.com/JRkjWRx.gif)
Aqui utilizei para novamente entender melhor os dados e suas aplicabilidades.

![Importância](https://i.imgur.com/FfH5RNK.png)

## Conclusão

Juntos, esses códigos formam um fluxo de trabalho robusto para a previsão de preços de venda de casas. Desde o pré-processamento e modelagem até a avaliação e análise exploratória, cada componente desempenha um papel crucial na geração de insights acionáveis e na melhoria da precisão das previsões.
