import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Carregar os dados processados
try:
    df = pd.read_csv('processed_data.csv')
    print("Dados processados carregados com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo processed_data.csv não encontrado. Execute o script de análise de dados primeiro.")
    exit()

# 2. Fazer One-Hot Encoding nas features categóricas
df_encoded = pd.get_dummies(df, columns=['Region', 'Energy_Source', 'Season'], drop_first=True)

# 3. Definir features (X) e target (y)
# Features a serem usadas (todas as numéricas + as categóricas codificadas)
features_to_drop = ['Energy_Class', 'Energy_Class_Encoded']
X = df_encoded.drop(columns=features_to_drop, errors='ignore')
y = df_encoded['Energy_Class_Encoded']

# 4. Padronizar as features numéricas
# Identificar colunas numéricas (excluindo as dummy e o target)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['Energy_Class_Encoded']]

# Aplicar a padronização (StandardScaler)
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Dicionário para armazenar resultados
results = {}

# Função para treinar e avaliar o modelo
def train_and_evaluate(X, y, test_size, scenario_name):
    print(f"\n--- Cenário: {scenario_name} (Test Size: {test_size*100}%) ---")
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Treinamento do modelo (Regressão Logística Multinominal - 'ovr' para multiclasse)
    # O solver 'lbfgs' é o padrão e suporta 'multinomial', mas o 'ovr' é mais explícito para o caso multiclasse
    # Para garantir a linearidade, usamos o LogisticRegression que é intrinsecamente linear
    model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_test)
    
    # Avaliação
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # Precision e Recall: usar 'weighted' para multiclasse desbalanceada
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Armazenar resultados
    results[scenario_name] = {
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'test_size': test_size
    }
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Matriz de Confusão:\n", cm)

# 5. Cenário 1: 40% dos dados para o conjunto de testes
train_and_evaluate(X, y, test_size=0.40, scenario_name="Cenario_40_percent")

# 6. Cenário 2: 15% dos dados para o conjunto de teste
train_and_evaluate(X, y, test_size=0.15, scenario_name="Cenario_15_percent")

# 7. Salvar os resultados em um arquivo Markdown
output_markdown = "## Resultados da Classificação com Regressão Logística\n\n"
output_markdown += "### Mapeamento de Classes\n"
output_markdown += "| Classe | Encoding Numérico |\n"
output_markdown += "|:---|:---:|\n"
output_markdown += "| High | 0 |\n"
output_markdown += "| Low | 1 |\n"
output_markdown += "| Medium | 2 |\n\n"

for scenario, data in results.items():
    output_markdown += f"### {scenario.replace('_', ' ').replace('percent', '%')}\n"
    output_markdown += f"**Tamanho do Conjunto de Teste:** {data['test_size']*100}%\n\n"
    
    output_markdown += "**Métricas de Performance:**\n"
    output_markdown += "| Métrica | Valor |\n"
    output_markdown += "|:---|:---:|\n"
    output_markdown += f"| Acurácia Geral | {data['accuracy']:.4f} |\n"
    output_markdown += f"| Precisão (Ponderada) | {data['precision']:.4f} |\n"
    output_markdown += f"| Recall (Ponderado) | {data['recall']:.4f} |\n\n"
    
    output_markdown += "**Matriz de Confusão:**\n"
    output_markdown += "As linhas representam as classes verdadeiras (True) e as colunas as classes preditas (Predicted).\n\n"
    
    # Criar tabela da Matriz de Confusão
    classes = ['High (0)', 'Low (1)', 'Medium (2)']
    cm_table = "| True/Predicted | " + " | ".join(classes) + " |\n"
    cm_table += "|:---|:---:|" + ":---:|" * (len(classes) - 1) + "\n"
    
    for i, row in enumerate(data['confusion_matrix']):
        cm_table += f"| {classes[i]} | " + " | ".join(map(str, row)) + " |\n"
    
    output_markdown += cm_table + "\n"

with open('avaliacao_modelo.md', 'w') as f:
    f.write(output_markdown)

print("\nResultados da avaliação salvos em 'avaliacao_modelo.md'.")
