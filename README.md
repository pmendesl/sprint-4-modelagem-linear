Integrantes: Pedro Mendes RM 562242 / Leonardo Augusto RM: 565564 / Alexandre RM: 563346 / Guilherme Peres RM: 563981 / Gabriel de Matos RM: 565218

# Resolução do Exercício de Classificação com Regressão Logística

Este repositório contém a resolução detalhada dos exercícios de classificação de Machine Learning, utilizando o dataset **Renewable Energy Production Dataset (2010-2020)**.

## 1. Análise de Dados e Matriz de Correlação

### a) Carregamento e Encoding do Target

O dataset foi carregado e a coluna alvo (`Energy_Class`) foi codificada para valores numéricos utilizando `LabelEncoder` para o treinamento do modelo.

| Classe | Encoding Numérico |
|:---|:---:|
| **High** | 0 |
| **Low** | 1 |
| **Medium** | 2 |

### b) Matriz de Correlação

A matriz de correlação de Pearson entre as features numéricas e a variável alvo codificada (`Energy_Class`) foi calculada.

**Matriz de Correlação (Resumo da Relação com o Target):**

| Feature | Correlação com Energy\_Class |
|:---|:---:|
| `Temperature_C` | 0.03 |
| `Wind_Speed_m_s` | 0.00 |
| `Solar_Radiation_kWh_m2` | -0.01 |
| `Rainfall_mm` | 0.05 |
| `Efficiency_Ratio` | -0.17 |
| `Lagged_Production_MWh` | 0.02 |
| `Combined_Weather_Index` | -0.01 |

### c) Escolha de Features

**Decisão:** É melhor **utilizar todas as features numéricas disponíveis**, além de codificar as features categóricas (`Region`, `Energy_Source`, `Season`).

**Justificativa:**

A correlação linear entre a variável alvo e todas as features numéricas é **muito baixa** (próxima de zero). A baixa correlação linear sugere que as features podem ter relações não-lineares ou que a classificação depende da combinação de múltiplas variáveis. Portanto, remover features com base apenas neste critério poderia resultar na perda de informações valiosas para o modelo de classificação.

## 2. Definição e Treinamento do Modelo

### a) Modelo de Classificação Linear

O modelo escolhido é a **Regressão Logística (`LogisticRegression`)** do `scikit-learn`.

### b) Por que a Regressão Logística é Linear?

A Regressão Logística é um classificador linear porque sua **fronteira de decisão (decision boundary) é linear**.

*   O modelo utiliza uma **combinação linear** das features de entrada para calcular a probabilidade de pertencer a uma classe.
*   A condição de decisão (onde a probabilidade é 50%) define um **hiperplano** no espaço de features, que é, por definição, uma estrutura linear.

### c) Classificação em Dois Cenários

O modelo foi treinado e avaliado utilizando todas as features (numéricas e categóricas codificadas) em dois cenários distintos de divisão de dados:

1.  **Cenário 40%:** 40% dos dados para o conjunto de testes.
2.  **Cenário 15%:** 15% dos dados para o conjunto de testes.

## 3. Avaliação da Performance

As métricas de performance (Acurácia, Precisão e Recall) e as Matrizes de Confusão foram calculadas para cada cenário.

### Resultados Consolidados

| Métrica | Cenário 40% (Teste) | Cenário 15% (Teste) |
|:---|:---:|:---:|
| **Acurácia Geral** | 0.6925 | 0.7600 |
| **Precisão (Ponderada)** | 0.6720 | 0.7573 |
| **Recall (Ponderado)** | 0.6925 | 0.7600 |

### Matrizes de Confusão

**Cenário 40% (Teste = 400 amostras):**

| True/Predicted | High (0) | Low (1) | Medium (2) |
|:---|:---:|:---:|:---:|
| **High (0)** | 191 | 0 | 4 |
| **Low (1)** | 12 | 28 | 31 |
| **Medium (2)** | 56 | 20 | 58 |

**Cenário 15% (Teste = 150 amostras):**

| True/Predicted | High (0) | Low (1) | Medium (2) |
|:---|:---:|:---:|:---:|
| **High (0)** | 72 | 0 | 1 |
| **Low (1)** | 2 | 13 | 12 |
| **Medium (2)** | 18 | 3 | 29 |

### Comparação e Justificativa (40% vs. 15%)

| Cenário | Vantagem | Desvantagem |
|:---|:---|:---|
| **Teste Menor (15%)** | **Mais dados para treino** (85%), resultando em um modelo mais robusto e métricas aparentemente melhores (Acurácia de 0.7600). | **Avaliação menos confiável** devido ao pequeno conjunto de teste (150 amostras), podendo ser uma estimativa otimista e com alta variância. |
| **Teste Maior (40%)** | **Avaliação mais confiável** e representativa da performance real do modelo. | **Menos dados para treino** (60%), o que pode resultar em um modelo ligeiramente sub-treinado e métricas mais baixas (Acurácia de 0.6925), mas mais fiéis à capacidade de generalização. |

**Conclusão:** O desempenho no Cenário 15% é superior, mas o resultado do Cenário 40% é considerado uma estimativa mais **realista e robusta** da performance do modelo em dados não vistos.

## 4. Arquivos Entregues

| Arquivo | Descrição |
|:---|:---|
| `Renewable_Energy_Data.xlsx` | Base de dados utilizada, incluindo as colunas de encoding. |
| `analise_dados.py` | Script Python para carregamento, encoding e cálculo da matriz de correlação. |
| `modelo_classificacao.py` | Script Python para treinamento do modelo de Regressão Logística e avaliação nos dois cenários. |
| `matriz_correlacao.md` | Matriz de correlação em formato Markdown. |
| `avaliacao_modelo.md` | Resultados detalhados da avaliação do modelo (Matrizes de Confusão e Métricas). |
| `heatmap_correlacao.png` | Visualização gráfica da matriz de correlação. |
