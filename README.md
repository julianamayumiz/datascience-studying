# Projeto 1: Previsão de Custos Médicos com Regressão Linear
 
Projeto de baseline em regressão supervisionada, parte da Fase 1 ("O Alicerce") de um plano de estudos em Data Science focado em métricas de erro, correlação, linearidade e interpretação de modelos.
 
## Objetivo
 
Prever o custo de seguro médico de uma pessoa a partir de características como idade, IMC, número de filhos, sexo, hábito de fumar e região de residência.
 
## Dataset
 
[Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance) (Kaggle) — 1.338 registros, 7 colunas (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).
 
## Ferramentas
 
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `scipy`
 
## Metodologia
 
1. **Limpeza:** verificação de nulos (nenhum encontrado) e remoção de 1 linha duplicada
2. **Análise exploratória (EDA):** distribuição das variáveis, boxplots por categoria, scatter plots com interação por `smoker`, mapa de correlação
3. **Preparação:** One-Hot Encoding para variáveis categóricas (`sex`, `smoker`, `region`)
4. **Modelagem:** Regressão Linear (scikit-learn) e OLS (statsmodels, para inferência estatística com p-valores)
5. **Avaliação:** métricas de erro (MAE, MSE, RMSE, R²) e diagnóstico de resíduos
## Principais achados
 
- **`smoker` é a variável dominante do dataset**: fumantes pagam, em média, quase 4x mais que não-fumantes (~R$31.700 vs. ~R$8.400)
- Correlação linear simples de `age`, `bmi` e `children` com `charges` é fraca isoladamente (0,07 a 0,30), mas todas são estatisticamente significativas (p < 0,05)
- O coeficiente de `smoker_yes` (~R$23.078) é, de longe, o maior peso do modelo
- **Diagnóstico de resíduos revelou 3 clusters distintos** (não-fumantes / fumantes com IMC mais baixo / fumantes com IMC ≥ ~30), evidenciando uma **interação entre `smoker` e `bmi`** que a regressão linear simples, por ser aditiva, não captura — o efeito do IMC no custo é muito mais forte entre fumantes do que entre não-fumantes
## Resultados
 
| Métrica | Valor |
|---|---|
| MAE | R$ 4.177,05 |
| MSE | 35.478.020,68 |
| RMSE | R$ 5.956,34 |
| R² | 0,8069 |
 
## Próximos passos
 
- Criar feature de interação (`bmi × smoker`) para capturar o efeito não-aditivo identificado nos resíduos
- Testar regularização (Lasso/Ridge) e modelos não-lineares (árvores) para comparação de performance
## Estrutura
 
```
01-custos-medicos/
├── Fase1.ipynb
├── insurance.csv
└── README.md
```
 


