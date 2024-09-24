import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados tratados, como se estivéssemos recebendo o roteiro final de um Block Buster
train_data_cleaned = pd.read_csv('train_data_cleaned.csv')

# Listar as colunas numéricas, como os personagens principais da nossa história
numerical_columns = train_data_cleaned.select_dtypes(include=[np.number]).columns.tolist()

# Aqui é o Art Attack. É hora de entender visualmente
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.hist(train_data_cleaned[col], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histograma de {col}')  # Vamos colocar um título para redação? "Como foram minhas férias..."
    plt.xlabel(col)  # Aqui é quem está sendo monitorado após utilizar o Tor
    plt.ylabel('Frequência')  # Quantas vezes acessou
    plt.savefig(f'Graficos/histograma_{col}.png')  # Imprimindo o fax
    plt.close()  # +1 concluído, -1 (para ser feito)

# Review sobre..
stats = train_data_cleaned[numerical_columns].describe()
print(stats)  # Públicou nas redes

# Calcular a Matrix de correlação apenas com colunas numéricas, para saber quais dão match
correlation_matrix = train_data_cleaned[numerical_columns].corr()

# Selecionar correlações com SalePrice, porque queremos descobrir quem brilha mais na noite
saleprice_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)

# Exibir as correlações, "Parabéns aos casais"
print(saleprice_corr)
