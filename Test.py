import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

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

# Preenchendo valores ausentes nas colunas numéricas com a mediana
for col in train_data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    median_value = train_data_cleaned[col].median()
    train_data_cleaned[col].fillna(median_value, inplace=True)

# Para colunas categóricas, substituindo valores ausentes pela moda
for col in train_data_cleaned.select_dtypes(include=['object']).columns:
    mode_value = train_data_cleaned[col].mode()[0]
    train_data_cleaned[col].fillna(mode_value, inplace=True)

# Salvando o dataset com um Look todo produzido
train_data_cleaned.to_csv('text_output/train_data_cleaned.csv', index=False)

# Listando as colunas numéricas
numerical_columns = train_data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_columns)

# Separando o conjunto de treino em X e y
X_train = train_data_cleaned.drop(['SalePrice'], axis=1)
y_train = np.log(train_data_cleaned['SalePrice'])  # Aplicando log na variável alvo

# Codificando variáveis categóricas no conjunto de treino
X_train_encoded = pd.get_dummies(X_train, drop_first=True)

# Padronizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Treinando o modelo XGBoost
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X_train_scaled, y_train)

# Calculando o MSE e o R² no conjunto de treino
y_train_pred = model.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Mean Squared Error (MSE) with XGBoost: {mse}")
print(f"Coefficient of Determination (R²) with XGBoost: {r2}")

# Agora vamos aplicar o modelo no arquivo test.csv

# Aplicando o mesmo pré-processamento no test.csv
test_data_encoded = pd.get_dummies(test_data, drop_first=True)

# Garantindo que o conjunto de teste tenha as mesmas colunas que o de treino
missing_cols = set(X_train_encoded.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0
test_data_encoded = test_data_encoded[X_train_encoded.columns]

# Padronizando os dados do conjunto de teste
X_test_scaled = scaler.transform(test_data_encoded)

# Fazendo as previsões no conjunto de teste
y_pred_test = model.predict(X_test_scaled)

# Convertendo as previsões de volta da escala logarítmica
y_pred_test_exp = np.exp(y_pred_test)

# Salvando as previsões em um arquivo CSV
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Usar a coluna 'Id' do arquivo test.csv
    'SalePrice': y_pred_test_exp
})

submission.to_csv('text_output/submission.csv', index=False)

print("Previsões salvas no arquivo submission.csv!")
