# Plano — Projeto 3: Regressão Não-Linear (Ensembles) — Previsão de Preços de Passagens Aéreas

## Visão Geral

**Objetivo:** Prever o preço de passagens aéreas a partir de características do voo (companhia, rota, classe, antecedência da compra, duração, escalas), explorando como modelos baseados em árvore capturam relações não-lineares que modelos lineares falham em capturar.

**Dataset:** [Flight Price Prediction (Kaggle)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) — dois arquivos: `Clean_Dataset.csv` (300k+ linhas, já processado) e dados brutos originais.

**Progressão pedagógica:**
```
Regressão Linear (baseline) → Random Forest → GradientBoosting (sklearn) → XGBoost
```
Cada salto revela *por que* o próximo modelo existe e o que ele resolve.

**Conceitos-Chave:**
- Não-linearidade e interações entre features (ex: `days_left` × `class`)
- Bagging vs. Boosting: diferença de filosofia e comportamento
- Feature Importance (MDI vs. Permutation Importance)
- Hiperparâmetros de árvore: `n_estimators`, `max_depth`, `learning_rate`
- Comparação estruturada de modelos: mesmas métricas, mesmo split

---

## Sub-Tarefas

---

### Sub-Tarefa 1 — Setup, Exploração da Raw Data e Entendimento do Domínio

**Status:** `[x] done`

**Intent:**
Criar a estrutura da pasta do projeto, baixar os dados e fazer uma exploração inicial da versão bruta do dataset para entender quais desafios de limpeza existem — antes de passar para o Clean Dataset. Isso conecta com o que foi feito nos projetos 1 e 2 (sempre partir do dado bruto primeiro).

**Expected Outcomes:**
- Pasta `03-passagens-aereas/` criada com notebook `Passagens Aereas.ipynb` e os arquivos CSV.
- Seção no notebook descrevendo o domínio do problema (o que cada coluna significa).
- Output de `df.head()`, `df.info()`, `df.describe()` e `df.isnull().sum()` rodando.
- Célula markdown explicando os desafios encontrados na Raw Data (colunas com unidades embutidas, formato de duração, datas etc.) e por que vamos usar o Clean Dataset para modelagem.

**Todo List:**
1. Criar pasta `datascience-studying/03-passagens-aereas/`.
2. Baixar o dataset do Kaggle (ambas as versões) e colocar na pasta.
3. Abrir o notebook; criar célula de título e contexto do projeto (tema, objetivo, conceitos-chave).
4. Carregar a Raw Data com pandas; rodar `head()`, `info()`, `describe()`, `isnull().sum()`.
5. Escrever célula markdown identificando os problemas da Raw Data: formato de `duration` ("2h 30m"), `dep_time`/`arr_time` como strings, `stops` como texto ("zero", "one", "2+"), `price` já limpo.
6. Carregar o Clean Dataset; mostrar as diferenças (colunas já numéricas, `days_left`, `duration` em horas float, `stops` codificado como inteiro).
7. Escrever célula markdown explicando a decisão: "usamos o Clean Dataset para modelagem, mas entendemos de onde cada feature veio".

**Relevant Context:**
- Padrão dos projetos anteriores: sempre abrir com `head()` + `info()` + `describe()` antes de qualquer limpeza.
- Arquivo Clean Dataset tem colunas: `airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`, `price`.

---

### Sub-Tarefa 2 — EDA: Explorando as Relações Não-Lineares

**Status:** `[x] done`

**Intent:**
Fazer uma EDA focada em *revelar visualmente* as não-linearidades que motivam o uso de ensembles. O foco são dois fenômenos centrais do domínio: (1) a relação curvilinear entre `days_left` e `price`, e (2) a interação entre `class` × `days_left` × `price`.

**Expected Outcomes:**
- Gráfico de dispersão/linha `days_left` × `price` mostrando a curva em "U" (preços altos muito próximos ao voo e muito distantes, mínimo intermediário).
- Boxplots de `price` por `airline`, `stops`, `class`.
- Heatmap de correlação das variáveis numéricas.
- Célula markdown conectando cada visualização ao conceito: "isso é não-linear, regressão linear vai falhar aqui porque..."

**Todo List:**
1. Importar `matplotlib`, `seaborn`.
2. Gráfico 1: lineplot ou scatter de `days_left` (eixo x) × `price` médio (eixo y), por classe (`Economy`/`Business`) — mostrar a curva em "U" para cada classe.
3. Gráfico 2: boxplots de `price` por `airline` (ordenados pela mediana).
4. Gráfico 3: boxplots de `price` por `stops`.
5. Gráfico 4: boxplots de `price` por `departure_time`.
6. Gráfico 5: heatmap de correlação (`duration`, `days_left`, `stops`, `price`).
7. Célula markdown de conclusões da EDA: quais relações são não-lineares, quais interações existem, o que esperamos que o modelo linear falhe em capturar.

**Relevant Context:**
- A relação `days_left` × `price` é o coração do projeto ("Desafio: Capturar relações complexas e não-lineares (ex: dias até o voo vs. preço)").
- Interações entre features (ex: classe Business tem dinâmica de preço diferente da Economy em função da antecedência) são exatamente o que árvores capturam nativamente.

---

### Sub-Tarefa 3 — Preparação dos Dados e Baseline Linear

**Status:** `[x] done`

**Intent:**
Preparar o dataset para modelagem (encoding de categóricas, split treino/teste) e treinar um modelo de Regressão Linear como baseline. O objetivo é *confirmar*, com números, que o modelo linear é insuficiente para este problema — criando a motivação concreta para os ensembles.

**Expected Outcomes:**
- Dataset encodado (OHE para categóricas com baixa cardinalidade: `airline`, `source_city`, `destination_city`, `departure_time`, `arrival_time`, `class`, `stops`).
- Split 80/20 com `random_state=42` (mesmo padrão dos projetos anteriores).
- Resultados do baseline linear: R² e RMSE no treino e teste, com gap de overfitting/underfitting visível.
- Gráfico Real vs. Previsto mostrando que o modelo linear subestima/superestima sistematicamente em faixas de preço.
- Célula markdown: "O modelo linear erra de forma sistemática em certos segmentos — isso é sinal de que a relação não é linear. É exatamente isso que motiva os modelos de árvore."

**Todo List:**
1. Definir quais colunas são features e qual é o target (`price`).
2. Aplicar One-Hot Encoding nas colunas categóricas (sem `drop_first` aqui, pois árvores não sofrem de dummy trap).
3. Fazer `train_test_split` 80/20 com `random_state=42`.
4. Treinar `LinearRegression` no treino.
5. Calcular MAE, RMSE, R² no treino e no teste.
6. Plotar Real vs. Previsto para o conjunto de teste.
7. Escrever célula markdown registrando os resultados e a interpretação do fracasso do modelo linear.

**Relevant Context:**
- Nos projetos anteriores, o OHE usou `drop_first=True` (para evitar dummy trap em modelos lineares). Para árvores, não é necessário — e documentar essa diferença é um ponto de aprendizado.
- A tabela de comparação final de todos os modelos começa aqui (criar uma variável/dicionário `resultados` para acumular métricas).

---

### Sub-Tarefa 4 — Random Forest: Bagging e Feature Importance

**Status:** `[x] done`

**Intent:**
Treinar um `RandomForestRegressor`, explicar o conceito de Bagging e como florestas reduzem variância, e introduzir Feature Importance como ferramenta de interpretabilidade.

**Expected Outcomes:**
- Modelo Random Forest treinado com parâmetros razoáveis (ex: `n_estimators=200`, `random_state=42`).
- Métricas (MAE, RMSE, R²) no treino e teste, mostrando melhora significativa vs. Linear.
- Gráfico de Feature Importance (top 15 features).
- Célula markdown explicando: o que é Bagging, por que a Random Forest reduz overfitting vs. uma única árvore profunda, o que a Feature Importance (MDI) mede e sua limitação (bias para features com alta cardinalidade).

**Todo List:**
1. Importar `RandomForestRegressor`.
2. Treinar com `n_estimators=200`, `random_state=42`, demais parâmetros default.
3. Calcular MAE, RMSE, R² treino/teste; adicionar ao dicionário `resultados`.
4. Extrair `feature_importances_` e plotar top 15 em gráfico de barras horizontal.
5. Célula markdown: conceito de Bagging (muitas árvores independentes em subsets aleatórios, resultado = média), por que reduz variância, limitação do MDI (Mean Decrease Impurity).

**Relevant Context:**
- Não é necessário escalar features para árvores (ao contrário de Ridge/Lasso do Projeto 2) — isso é um ponto a documentar.
- `feature_importances_` é o MDI; para uma análise mais honesta, seria usado `permutation_importance`, mas MDI é suficiente para este projeto introdutório.

---

### Sub-Tarefa 5 — GradientBoosting (sklearn): Introduzindo o Boosting

**Status:** `[x] done`

**Intent:**
Treinar um `GradientBoostingRegressor` do sklearn para introduzir o conceito de Boosting — aprendizado sequencial de erros. Comparar com Random Forest para mostrar a diferença de filosofia entre Bagging e Boosting.

**Expected Outcomes:**
- Modelo GBM treinado (`n_estimators=200`, `learning_rate=0.1`, `max_depth=4`, `random_state=42`).
- Métricas no treino e teste; comparação com RF.
- Gráfico de curva de aprendizado (`train_score` vs `test_score` em função de `n_estimators`) usando `staged_predict`.
- Célula markdown: diferença conceitual Bagging × Boosting, o papel de `learning_rate` ("shrinkage"), risco de overfitting com `n_estimators` alto, por que GBM tende a ser mais preciso porém mais lento que RF.

**Todo List:**
1. Importar `GradientBoostingRegressor`.
2. Treinar com parâmetros descritos acima.
3. Calcular MAE, RMSE, R² treino/teste; adicionar ao `resultados`.
4. Usar `staged_predict` para plotar a curva de aprendizado (RMSE vs. número de árvores, treino e teste na mesma figura).
5. Célula markdown: conceito de Boosting (ajuste iterativo dos resíduos), papel do `learning_rate`, por que a curva de aprendizado mostra se há overfitting.

**Relevant Context:**
- `staged_predict` é um método nativo do `GradientBoostingRegressor` que retorna previsões após cada estágio — permite plotar a curva de aprendizado sem retreinar.
- A curva de aprendizado é um diagnóstico visual poderoso para escolher `n_estimators` ideal (ponto onde a curva de teste para de melhorar).

---

### Sub-Tarefa 6 — XGBoost: O Estado da Arte

**Status:** `[x] done`

**Intent:**
Treinar um modelo XGBoost, mostrar sua superioridade prática (velocidade + performance) sobre o GBM clássico, e introduzir o conceito de regularização nativa (L1/L2) em modelos de árvore.

**Expected Outcomes:**
- Modelo XGBoost treinado (`n_estimators=300`, `learning_rate=0.1`, `max_depth=6`, `random_state=42`).
- Métricas no treino e teste, geralmente o melhor resultado da comparação.
- Feature Importance do XGBoost (`plot_importance`).
- Célula markdown: o que o XGBoost adiciona ao GBM clássico (regularização L1/L2 nativa, tratamento de missing values, paralelismo na construção das árvores), por que é o modelo padrão em competições Kaggle.

**Todo List:**
1. Instalar/importar `xgboost`.
2. Treinar `XGBRegressor` com parâmetros descritos.
3. Calcular MAE, RMSE, R² treino/teste; adicionar ao `resultados`.
4. Plotar Feature Importance com `xgboost.plot_importance`.
5. Célula markdown explicando as melhorias do XGBoost sobre o GBM clássico.

**Relevant Context:**
- XGBoost pode receber os dados sem OHE se usar `enable_categorical=True`, mas para consistência com os outros modelos, vamos usar o mesmo dataset encodado.
- O parâmetro `tree_method='hist'` acelera muito o treinamento em datasets grandes — vale documentar.

---

### Sub-Tarefa 7 — Comparação Final e Conclusões

**Status:** `[x] done`

**Intent:**
Consolidar os resultados de todos os modelos em uma tabela comparativa, gerar visualizações finais e extrair as conclusões de aprendizado do projeto — conectando os resultados de volta às não-linearidades identificadas na EDA.

**Expected Outcomes:**
- Tabela comparativa: `Modelo | MAE Treino | RMSE Treino | R² Treino | MAE Teste | RMSE Teste | R² Teste`.
- Gráfico de barras comparando RMSE no teste para todos os modelos.
- Gráfico Real vs. Previsto para o melhor modelo (XGBoost), mostrando a melhora vs. o baseline linear.
- Célula markdown de conclusões conectando cada resultado aos conceitos-chave do projeto.
- Atualização do `README.md` do projeto e do `README.md` raiz (marcar Projeto 3 como concluído na tabela).

**Todo List:**
1. Criar DataFrame `df_resultados` a partir do dicionário `resultados` acumulado.
2. Exibir `df_resultados` formatado como tabela.
3. Plotar gráfico de barras: RMSE no teste por modelo (4 barras).
4. Plotar Real vs. Previsto do XGBoost no teste.
5. Escrever célula markdown de conclusões: qual modelo ganhou, por que, o que cada conceito aprendido explica o resultado.
6. Criar `README.md` da pasta `03-passagens-aereas/` seguindo o padrão do Projeto 1 (objetivo, dataset, ferramentas, metodologia, resultados, próximos passos).
7. Atualizar `datascience-studying/README.md`: adicionar linha do Projeto 3 na tabela de projetos.

**Relevant Context:**
- Padrão de README: ver `01-custos-medicos/README.md` como template.
- A tabela raiz do README só marca projetos com ✅ após conclusão.
