# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# "É logo ali..." 
train_file_path = 'train.csv'  # Corrigido para usar train.csv

# Seleção Eletro Hits (só as melhores)
train_data = pd.read_csv(train_file_path)

# Criando mashups, remixes e remakes de tracks que já existiam
train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']

# Análise e geração de texto
with open('text_output/analise_variaveis.txt', 'w') as f:  # Ajustado para text_output
    f.write("Análise das Variáveis\n")
    f.write("====================\n\n")
    
    # Interação entre TotalSF e SalePrice
    f.write("1. Interação entre Total SF e Preço de Venda:\n")
    f.write(f"Média do Preço de Venda para Total SF:\n{train_data.groupby('TotalSF')['SalePrice'].mean()}\n\n")
    
    # Interação entre OverallQual e SalePrice
    f.write("2. Interação entre Qualidade Geral e Preço de Venda:\n")
    f.write(f"Média do Preço de Venda por Qualidade Geral:\n{train_data.groupby('OverallQual')['SalePrice'].mean()}\n\n")
    
    # Interação entre GarageCars e SalePrice
    f.write("3. Interação entre Número de Garagens e Preço de Venda:\n")
    f.write(f"Média do Preço de Venda por Número de Garagens:\n{train_data.groupby('GarageCars')['SalePrice'].mean()}\n\n")

# VAR , pois não podemos ter dúvida do que aconteceu
plt.figure(figsize=(12, 10))
sns.scatterplot(x=train_data['TotalSF'], y=train_data['SalePrice'])
plt.title('Interação entre Total SF e Preço de Venda') 
plt.xlabel('Total SF') 
plt.ylabel('Preço de Venda')  
plt.savefig('Graficos/interacao_total_sf_sale_price.png')  # Ajustado para Graficos
plt.close()

plt.figure(figsize=(12, 10))
sns.boxplot(x=train_data['OverallQual'], y=train_data['SalePrice'])
plt.title('Interação entre Qualidade Geral e Preço de Venda')  
plt.xlabel('Qualidade Geral')  
plt.ylabel('Preço de Venda')  
plt.savefig('Graficos/interacao_overall_quality_sale_price.png')  # Ajustado para Graficos
plt.close()

plt.figure(figsize=(12, 10))
sns.boxplot(x=train_data['GarageCars'], y=train_data['SalePrice'])
plt.title('Interação entre Número de Garagens e Preço de Venda') 
plt.xlabel('Número de Garagens')  
plt.ylabel('Preço de Venda') 
plt.savefig('Graficos/interacao_garage_cars_sale_price.png')  # Ajustado para Graficos
plt.close()
