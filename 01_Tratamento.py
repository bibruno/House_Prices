import pandas as pd
import warnings
import numpy as np

# Aqui estou ignorando os gritos de aviso do Python 
warnings.filterwarnings('ignore')


# GPS "...vire a direita em 200m..."
train_file_path = 'train.csv'
test_file_path = 'test.csv'


# Quem é quem na fila do pão
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Exibindo as informações gerais sobre o dataset, como um relatório de saúde
print(train_data.info())

# Sinopse da série
print(train_data.describe())

# Checando se há valores ausentes, "Onde está Carmen San Diego?"
missing_values = train_data.isnull().sum()
print(missing_values[missing_values > 0])

# Removendo colunas com mais de 50% de valores ausentes, porque não precisamos de convidados indesejados
threshold = len(train_data) * 0.5
train_data_cleaned = train_data.dropna(axis=1, thresh=threshold)

# Preenchendo valores ausentes nas colunas numéricas com a mediana, como um bom chefe de cozinha que sabe como equilibrar os sabores
for col in train_data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    median_value = train_data_cleaned[col].median()
    train_data_cleaned[col].fillna(median_value, inplace=True)

# Para colunas categóricas, substituindo valores ausentes pela moda, como se estivesse escolhendo a camiseta mais popular da festa
for col in train_data_cleaned.select_dtypes(include=['object']).columns:
    mode_value = train_data_cleaned[col].mode()[0]
    train_data_cleaned[col].fillna(mode_value, inplace=True)

# Salvando o dataset com um Look todo produzido
train_data_cleaned.to_csv('text_output/train_data_cleaned.csv', index=False)

# Listando as colunas numéricas, para já sabermos quem vai dar trabalho... (ou não)
numerical_columns = train_data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_columns)
