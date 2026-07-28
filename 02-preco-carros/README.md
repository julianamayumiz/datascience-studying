# Regressão com Regularização — Precificação de Veículos Usados

Projeto de estudo em Machine Learning com foco em Regressão Linear, Ridge e Lasso, aplicado à previsão do preço de revenda de veículos usados.

## 🎯 Objetivo

Prever o preço de revenda de veículos usados a partir de características técnicas (potência, motor, quilometragem, ano) e categóricas (marca, combustível, transmissão, localização), comparando Regressão Linear simples com as versões regularizadas Ridge (L2) e Lasso (L1).

## 📊 Dataset

[Used Car Price Prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction) (Kaggle) — 6.019 registros de carros usados, com colunas como marca/modelo, ano, quilometragem, tipo de combustível, transmissão, motor, potência e preço.

## 🧩 Desafio principal

Lidar com múltiplas variáveis categóricas de alta cardinalidade (marca, modelo) sem inflar demais a dimensionalidade do problema, e entender como a regularização se comporta diante de multicolinearidade e overfitting.

## 🔧 Pipeline do projeto

1. **EDA e limpeza básica**
   - Remoção de colunas inúteis (`Unnamed: 0`) e com excesso de nulos (`New_Price`, ~86% ausente).
   - Extração da marca (`Brand`) a partir da coluna `Name`, reduzindo de ~2.000 nomes únicos para 31 marcas — evitando alta cardinalidade no encoding.
   - Conversão de `Mileage`, `Engine` e `Power` de texto (ex: `"26.6 kmpl"`) para numérico.
   - Imputação de nulos remanescentes (mediana para numéricas, moda para `Seats`).

2. **Pré-processamento**
   - One-Hot Encoding (`pd.get_dummies`, `drop_first=True`) nas colunas categóricas (`Location`, `Fuel_Type`, `Transmission`, `Owner_Type`, `Brand`), evitando a dummy variable trap.
   - Dataset final: 55 colunas (após o encoding).

3. **Regressão Linear (baseline)**
   - Split treino/teste 80/20.
   - Evidenciou overfitting (R² treino 0.78 vs. teste 0.58) e coeficientes explosivos em marcas raras (ex: `Brand_Lamborghini` = +63.9) — sintoma de multicolinearidade em dummies esparsas.

4. **Ridge e Lasso**
   - Primeira rodada sem padronização: regularização praticamente ineficaz (alpha ótimo preso na borda da grade de busca).
   - Com `StandardScaler` (fit apenas no treino, aplicado no teste): resultados mais coerentes, com o Lasso superando o Ridge por eliminar features irrelevantes (seleção de variáveis).

5. **Investigação de outlier**
   - Análise de resíduos revelou um registro com 6.500.000 km rodados — erro de digitação, não outlier legítimo.
   - Remoção dessa única linha elevou o R² de teste de 0.59 para **0.73**, um ganho muito maior do que qualquer ajuste de regularização isolado.

6. **Visualizações**
   - Gráfico de coeficientes do Lasso (importância e direção do efeito de cada variável).
   - Real vs. Previsto.
   - Resíduos vs. Previsto (revelou heterocedasticidade — erro crescente em carros mais caros).

## 📈 Resultados

| Etapa | R² (teste) | RMSE (teste) |
|---|---|---|
| Regressão Linear (baseline) | 0.5797 | 7.19 |
| Ridge (sem padronização) | 0.5784 | 7.20 |
| Lasso (sem padronização) | 0.5628 | 7.34 |
| Ridge (padronizado) | 0.5746 | 7.24 |
| Lasso (padronizado) | 0.5918 | 7.09 |
| **Lasso (padronizado + outlier removido)** | **0.7328** | **5.97** |

## 💡 Principais aprendizados

- **One-Hot Encoding com `drop_first=True`** evita a dummy variable trap, mas variáveis de alta cardinalidade (`Brand`) amplificam multicolinearidade.
- **Regularização exige padronização.** Sem `StandardScaler`, Ridge e Lasso penalizam features de forma desigual conforme sua escala original, tornando a regularização praticamente sem efeito.
- **Ridge e Lasso resolvem problemas diferentes**: Ridge distribui peso entre variáveis correlacionadas; Lasso seleciona/descarta features irrelevantes (zera coeficientes). Neste dataset, o Lasso teve vantagem por lidar melhor com dummies pouco informativas.
- **Regularização não substitui qualidade de dado.** Um único valor implausível (erro de digitação) causou mais dano ao modelo do que qualquer ausência de regularização — e sua correção gerou o maior ganho de performance do projeto.
- **Overfitting tem causas distintas**: complexidade do modelo (resolvida por regularização) vs. contaminação por dados errados (resolvida só por limpeza). Saber diferenciar evita usar a ferramenta errada para o problema.
- **Limitação identificada**: heterocedasticidade persistente nos resíduos, com o modelo errando mais em carros caros — reflexo da distribuição assimétrica do preço. Transformação logarítmica de `Price` (`log(Price)`) foi identificada como próximo passo natural, não implementada neste ciclo.

## ✅ Conceitos dominados

- One-Hot Encoding e dummy variable trap
- Multicolinearidade (diagnóstico via coeficientes)
- Overfitting vs. Underfitting (gap treino-teste)
- Regularização L1 (Lasso) vs. L2 (Ridge)
- Padronização (`StandardScaler`) como pré-requisito da regularização
- Investigação de outliers via análise de resíduos

## 🔮 Próximos passos (não implementados)

- Testar transformação `log(Price)` para lidar com a assimetria da variável-alvo.
- Investigar os outros 3 casos de quilometragem extrema (620k–775k km) — não removidos, por serem plausíveis (frota/táxi).

## 🛠️ Stack

Python · Pandas · NumPy · Scikit-Learn (`LinearRegression`, `RidgeCV`, `LassoCV`, `StandardScaler`) · Matplotlib
