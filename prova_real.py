import pandas as pd

# Caminhos para os arquivos CSV
train_file_path = '/home/bruno/Documents/ML_STUDY/House Prices/train.csv'
train_cleaned_file_path = '/home/bruno/Documents/ML_STUDY/House Prices/train_data_cleaned.csv'
test_file_path = '/home/bruno/Documents/ML_STUDY/House Prices/test.csv'

# Lendo os arquivos
train_data = pd.read_csv(train_file_path)
train_data_cleaned = pd.read_csv(train_cleaned_file_path)
test_data = pd.read_csv(test_file_path)

# Número de linhas
num_rows_train = len(train_data)
num_rows_train_cleaned = len(train_data_cleaned)
num_rows_test = len(test_data)

# Cálculo da porcentagem
percentage_cleaned = (num_rows_train_cleaned / num_rows_train) * 100
percentage_test = (num_rows_train_cleaned / num_rows_test) * 100

# Resultados
print(f"Número de linhas em train_data_cleaned: {num_rows_train_cleaned}")
print(f"Número de linhas em train.csv: {num_rows_train}")
print(f"Número de linhas em test.csv: {num_rows_test}")
print(f"Isso representa {percentage_cleaned:.2f}% do conjunto de treinamento.")
print(f"Isso representa {percentage_test:.2f}% do conjunto de teste.")
