Readme · MD
# Projeto 3: Previsão de Preços de Passagens Aéreas com Ensembles

Projeto de regressão não-linear, parte da progressão de estudos em Data Science focada em modelos baseados em árvore (Random Forest, Gradient Boosting, XGBoost) e comparação com regressão linear.

## Objetivo

Prever o preço de passagens aéreas a partir de características do voo (companhia, rota, classe, antecedência da compra, duração, escalas), explorando como modelos baseados em árvore capturam relações não-lineares que modelos lineares falham em capturar.

## Dataset

[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (Kaggle) — 300k+ registros de voos na Índia. Usados dois formatos: a raw data (`business.csv` + `economy.csv`, separada por classe) e o `Clean_Dataset.csv` (já tratado, usado para a modelagem).

## Ferramentas

`pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

## Metodologia

1. **Exploração da raw data:** identificação de problemas de formatação (`price` com vírgula de milhar, `stop` com lixo de formatação, `date`/`time_taken`/`dep_time`/`arr_time` em formatos não-numéricos, índices duplicados após concatenação) — decisão de usar o Clean Dataset para modelagem
2. **EDA:** relação `days_left` × `price`, boxplots por `airline`/`stops`/`departure_time`, heatmap de correlação
3. **Preparação:** One-Hot Encoding das categóricas (sem `drop_first`, já que árvores não sofrem de dummy trap), split 80/20 (`random_state=42`)
4. **Modelagem:** Regressão Linear (baseline) → Random Forest (Bagging) → Gradient Boosting (Boosting) → XGBoost (Boosting regularizado)
5. **Avaliação:** MAE, RMSE, R² em treino/teste para todos os modelos, gráficos Real vs. Previsto, Feature Importance (MDI e `weight`)

## Principais achados

- A relação `days_left` × `price` é fortemente **não-linear** (curva de decaimento acentuado seguida de platô), mas sua **correlação de Pearson é quase nula (-0,09)** — prova concreta de que correlação linear simples não detecta esse padrão, e de que a regressão linear tende a ignorá-lo
- `airline` e `stops` têm efeitos **contaminados pela interação com `class`**: `Air_India`/`Vistara` (únicas com Business) mostram variação de preço muito maior, e `one stop` tem mediana de preço maior que `zero stops` só porque tem mais proporção de assentos Business
- A Regressão Linear (R² 0,9099) captura bem o efeito grosseiro de `class`, mas comprime sistematicamente as previsões de voos Business numa faixa estreita, ignorando sua variação real (visível no gráfico Real vs. Previsto)
- **Random Forest foi o melhor modelo** (R² 0,9850 no teste), superando inclusive o XGBoost — resultado que contraria a expectativa de que Boosting regularizado venceria; a causa provável é a falta de tuning de hiperparâmetros, não uma limitação do XGBoost em si
- Feature Importance divergiu entre métricas: MDI (Random Forest) destacou `class` isoladamente por impacto de grande magnitude; `weight` (XGBoost) destacou `duration` por frequência de uso — nenhuma métrica conta a história completa sozinha

## Resultados

| Modelo | MAE Teste | RMSE Teste | R² Teste |
|---|---|---|---|
| Regressão Linear | 4.500,71 | 6.814,94 | 0,9099 |
| **Random Forest** | **1.075,49** | **2.782,56** | **0,9850** |
| Gradient Boosting | 2.618,50 | 4.488,05 | 0,9609 |
| XGBoost | 1.960,02 | 3.462,81 | 0,9767 |

## Próximos passos

- Tuning de hiperparâmetros (`GridSearchCV`/`RandomizedSearchCV`) para verificar se XGBoost supera o Random Forest com ajuste adequado
- `permutation_importance` para uma leitura mais robusta da relevância das features
- Engenharia de features de interação explícitas (ex: `airline` × `class`) para uma comparação mais justa com a Regressão Linear

## Estrutura

```
03-passagens-aereas/
├── Passagens Aereas.ipynb
├── Clean_Dataset.csv
├── business.csv
├── economy.csv
├── plano-projeto3.md
└── README.md
```
