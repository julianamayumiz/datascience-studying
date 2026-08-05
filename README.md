# Data Science — Projetos de Estudo

Repositório de projetos práticos de um plano de estudos em Data Science, organizado em fases progressivas — do fundamento estatístico (regressão) até tópicos mais avançados (classificação, ensemble, não-supervisionado).

## Estrutura do plano

### Fase 1: O Alicerce (Regressão e Business Understanding)
Foco: métricas de erro, correlação, linearidade e interpretação de modelos.

| Projeto | Tema | Status | Principais conceitos |
|---|---|---|---|
| [01 - Custos Médicos](./01-custos-medicos) | Previsão de custo de seguro médico | ✅ Concluído | Correlação, p-valor, coeficientes, RMSE/R², diagnóstico de resíduos |
| [02 - Precificação de Veículos Usados](./02-precificacao-veiculos-usados) | Previsão de preço de revenda de carros usados | ✅ Concluído | One-Hot Encoding, multicolinearidade, overfitting vs. underfitting, Regularização L1/L2 (Ridge/Lasso), padronização, investigação de outliers |
| [03 - Preços de Passagens Aéreas](./03-passagens-aereas) | Previsão de preço de passagens aéreas | ✅ Concluído | Regressão não-linear, Random Forest Regressor, XGBoost, comparação árvore vs. linear |

## Ferramentas utilizadas

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `scipy`

## Como navegar

Cada projeto tem sua própria pasta com notebook (`.ipynb`), dataset e README específico detalhando objetivo, metodologia e resultados.
