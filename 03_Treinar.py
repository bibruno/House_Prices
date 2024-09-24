import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Censurando o Python 
warnings.filterwarnings('ignore')

# "I can only show you the door. You're the one that has to walk through it"
train_file_path = 'train_data_cleaned.csv'

# Standardize features by removing the mean and scaling to unit variance. (de acordo com a doc)
scaler = StandardScaler()

# Aqui começa o caminho de tijolos dourados
train_data = pd.read_csv(train_file_path)

# Isso é tão lindo! Ele transforma classificações em colunas com 0 e 1 (resumidamente)
train_data_encoded = pd.get_dummies(train_data, drop_first=True)

# Aqui o caldo engrossa, selecionamos (com ajuda dos outros códigos) as variáveis que mais influenciam o preço e as tratamos como merecem: 
train_data_encoded['TotalSF'] = train_data_encoded['TotalBsmtSF'] + train_data_encoded['1stFlrSF'] + train_data_encoded['2ndFlrSF']
train_data_encoded['TotalBathrooms'] = train_data_encoded['FullBath'] + (train_data_encoded['HalfBath'] * 0.5) + train_data_encoded['BsmtFullBath'] + (train_data_encoded['BsmtHalfBath'] * 0.5)
train_data_encoded['TotalQuartos'] = train_data_encoded['BedroomAbvGr']
train_data_encoded['QualidadeGerais'] = train_data_encoded['OverallQual']
train_data_encoded['TotalGaragens'] = train_data_encoded['GarageCars']
train_data_encoded['Qualidade_TotalSF'] = train_data_encoded['TotalSF'] * train_data_encoded['OverallQual']
train_data_encoded['Garagens_SalePrice'] = train_data_encoded['TotalGaragens'] * train_data_encoded['SalePrice']
train_data_encoded['SalePrice'] = np.log(train_data_encoded['SalePrice'])

# Aqui é onde entra "NORMALIZAÇÃO" e dita o que é aceitável para o preço de venda (saudades six sigma)
Q1 = train_data_encoded['SalePrice'].quantile(0.25)
Q3 = train_data_encoded['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# E lá se vão os valores de preço de vendas fora da "NORMALIZAÇÃO" 
train_data_cleaned = train_data_encoded[~((train_data_encoded['SalePrice'] < limite_inferior) | (train_data_encoded['SalePrice'] > limite_superior))]

# Agora todas colunas exceto a SalePrice estão prontas para guiarem o treinamento para se obter: SalePrice
X = train_data_cleaned.drop(columns=['SalePrice'])
y = train_data_cleaned['SalePrice']

# Aqui a gente separa uma fatia dos dados (20%) para efetuar os testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aí, a gente monta uma escala para que números grandes não impressionem pelo tamanho (size matters?) mas mantendo a proporção
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# E agora o belholder acordou
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Aqui , basicamente é a ficha do Beholder, quantos olhos ele tem, quantos feitiços lança por turno e etc
param_grid = {
    'n_estimators': [10000],
    'max_depth': [5],
    'learning_rate': [0.003],
    'subsample': [0.3]  
}

# n_jobs=-1 (sidekick), scoring (seus pontos no final da fase) e cv=5 (vidas por fase)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, 
                           verbose=1, n_jobs=-1)

# Who Wants to Live Forever (só sobrevive o melhor)
grid_search.fit(X_train_scaled, y_train)

# Aqui já é podium com champagne (tá ouvindo a musiquinha né? pan pan pan)
best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)

# E FINALMENTE nossa bola de cristal funcionou...
y_pred_xgb = best_model.predict(X_test_scaled)

# Os jurados avaliam o prato e dizem se falta tômpero 
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# A gente guarda de recordação os valores que obtivemos (pra postar no insta)
with open('text_output/resultados_xgboost.txt', 'w') as f:  # Alterado para salvar em 'text_output'
    f.write(f"Mean Squared Error (MSE) with XGBoost: {mse_xgb}\n")
    f.write(f"Coefficient of Determination (R²) with XGBoost: {r2_xgb}\n")

# Só garantindo que não temos ninguém fora do padrão (Esquadrão da Moda)
outliers = train_data_cleaned[(train_data_cleaned['SalePrice'] < limite_inferior) | (train_data_cleaned['SalePrice'] > limite_superior)]
print(f"Number of identified outliers: {len(outliers)}")
print(outliers[['SalePrice']])  # Display the identified outliers

# Não entendeu nada? Então a gente desenha....
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title('Predicted vs Actual Sale Prices')
plt.xlabel('Actual Sale Price (log-transformed)')
plt.ylabel('Predicted Sale Price (log-transformed)')
plt.grid()
plt.savefig('Graficos/predicted_vs_actual.png')  # Alterado para salvar em 'Graficos'
plt.show()

# Já praticou tiro? Essa é a hora que a folha de papel vem para frente e você analisa sua performance
errors = y_test - y_pred_xgb
plt.figure(figsize=(12, 6))
sns.histplot(errors, bins=30, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--')
plt.grid()
plt.savefig('Graficos/prediction_errors.png')  # Alterado para salvar em 'Graficos'
plt.show()

# Quem foram os protagonistas, coadjuvante e etc...(para chegarmos aos resultados)
plt.figure(figsize=(12, 6))
xgb.plot_importance(best_model, 
                    max_num_features=10, 
                    importance_type='weight', 
                    xlabel='F score', 
                    ylabel='Features')
plt.title('Feature Importances from XGBoost Model')
plt.xticks(rotation=45)  # Rotate feature names for better readability
plt.savefig('Graficos/feature_importances.png')  # Alterado para salvar em 'Graficos'
plt.show()
