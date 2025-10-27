import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados
try:
    df = pd.read_csv('/home/ubuntu/upload/Renewable_Energy_Data.csv')
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo Renewable_Energy_Data.csv não encontrado.")
    exit()

# Exibir informações iniciais e as primeiras linhas
print("\nInformações do DataFrame:")
df.info()
print("\nPrimeiras 5 linhas:")
print(df.head())

# 2. Fazer o encoding do target (Energy_Class) para uma variável numérica
# 2.1 Verificar os valores únicos da coluna target
print("\nValores únicos em 'Energy_Class':", df['Energy_Class'].unique())

# 2.2 Aplicar Label Encoding
le = LabelEncoder()
df['Energy_Class_Encoded'] = le.fit_transform(df['Energy_Class'])

# Mapeamento para referência futura
encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nMapeamento de Encoding para 'Energy_Class':", encoding_map)

# 3. Montar a matriz de correlação
# Excluir a coluna original de string e outras colunas não numéricas se houver
df_numeric = df.select_dtypes(include=np.number)
df_numeric = df_numeric.drop(columns=['Year'], errors='ignore') # Assumindo que 'Year' não é uma feature para classificação
df_numeric = df_numeric.drop(columns=['Energy_Class'], errors='ignore') # A coluna original de string

# Renomear o target para o nome original para a matriz de correlação
df_numeric = df_numeric.rename(columns={'Energy_Class_Encoded': 'Energy_Class'})

correlation_matrix = df_numeric.corr()

# 4. Salvar a matriz de correlação em um arquivo Markdown
correlation_markdown = "## Matriz de Correlação\n\n"
correlation_markdown += correlation_matrix.to_markdown()

with open('matriz_correlacao.md', 'w') as f:
    f.write(correlation_markdown)

# 5. Gerar e salvar um heatmap da matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap da Matriz de Correlação')
plt.savefig('heatmap_correlacao.png')
print("\nMatriz de correlação e Heatmap salvos.")

# Salvar o DataFrame processado para uso nas próximas etapas
df.to_csv('processed_data.csv', index=False)
print("Dados processados salvos em 'processed_data.csv'.")

# Exibir o nome das features (colunas numéricas, exceto o target original)
features = df_numeric.drop(columns=['Energy_Class']).columns.tolist()
print("\nFeatures para classificação (incluindo o target codificado):")
print(df_numeric.columns.tolist())
print("\nScript concluído.")
