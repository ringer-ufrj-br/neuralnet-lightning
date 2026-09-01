# neuralnet-lightning

Orquestrador e pipeline de treinamento de redes neurais (baseado em PyTorch / PyTorch Lightning) voltado para análise de dados do ATLAS (CERN).

---

## 📌 Visão Geral

O projeto automatiza o fluxo de carregamento de dados (ex: arquivos Parquet), pré-processamento, criação e treinamento de modelos de deep learning (`MLP`, `CNN2D`), validação cruzada por K-Fold e avaliação de desempenho — incluindo o **tabelão** de validação cruzada em LaTeX.

O fluxo é dividido em **três comandos independentes**:

| Comando | O que faz | O que produz |
|---|---|---|
| `train` | Treina os folds da validação cruzada | Checkpoints, preprocessador ajustado, índices do holdout, manifesto |
| `evaluate` | Reinferência dos folds sobre o holdout | Scores, métricas por fold, gráficos, fatia do tabelão daquela região |
| `report` | Agrega todas as regiões avaliadas, de um ou vários modelos | Tabelão em `.tex`, figura da tabela e o CSV longo canônico |

A separação existe para que **re-avaliar não exija retreinar**: recortar pontos de operação, refazer gráficos ou remontar a tabela lê apenas artefatos em disco.

---

## 📁 Estrutura Principal do Projeto

- **`ai/`**: módulos de inteligência artificial.
  - `ai/run.py`: entrypoint com os subcomandos `train` / `evaluate` / `report`.
  - `ai/pipeline/base.py`: pipeline compartilhado (treino, avaliação, persistência de artefatos).
  - `ai/pipeline/pipeline_mlp.py`, `pipeline_cnn2d.py`: só declaram modelo e preprocessador.
  - `ai/models/`: arquiteturas das redes.
  - `ai/preprocess/`: preprocessadores (contrato `fit`/`transform`/`save`/`load`).
  - `ai/evaluation/`: métricas, gráficos e o construtor do tabelão (`tabelao.py`).
  - `ai/binning/kinematics.py`: bins de $E_T$ e $|\eta|$ (grade 5×5 = 25 redes).
- **`ai/configs/*.yaml`**: configurações e hiperparâmetros de cada experimento.
- **`data/`**: conjuntos de dados (Parquet/ROOT).
- **`results/`**: relatórios, métricas, gráficos e checkpoints.
- **`Makefile` & `activate.sh`**: utilitários de ambiente.

---

## ⚙️ Pré-requisitos e Instalação

1. **Criar o ambiente virtual e instalar as dependências:**
   ```bash
   make venv
   ```
2. **Ativar o ambiente virtual:**
   ```bash
   source activate.sh
   ```

---

## ⚙️ Configuração (`ai/configs/*.yaml`)

```yaml
model: "MLP"                       # Modelo a ser utilizado (MLP | CNN2D)
data_path: data/parquet/           # Caminho para os dados
max_files: 100                     # Quantidade máxima de arquivos por pasta
label_col: "label"                 # Coluna de rótulo
max_epochs: 50                     # Número máximo de épocas
batch_size: 128                    # Tamanho do batch
learning_rate: 0.001               # Taxa de aprendizado
patience: 8                        # Paciência do Early Stopping
n_splits: 5                        # Folds da validação cruzada (1 = sem validação cruzada)
test_size: 0.15                    # Proporção do holdout de teste
threshold: 0.8                     # Limiar fixo das métricas globais
seed: 42                           # Semente do holdout e da partição de folds
```

Os pontos de operação do tabelão são **tight 90% / medium 95% / loose 99%** por padrão, sem
precisar declarar nada. Para usar outros, sobrescreva no YAML (nome → $P_D$ alvo):

```yaml
operating_points:
  tight: 0.90
  medium: 0.95
  veryloose: 0.995
```

> `train` sempre roda validação cruzada. Para treinar um modelo só, use `n_splits: 1` —
> nesse caso o treino usa um único split estratificado de validação e o tabelão sai com um
> fold e desvio zero.

---

## 🚀 Como Executar

### 1. Treino

```bash
python ai/run.py train --config ai/configs/mlp.yaml
```

Um fold específico (paralelização em SLURM):

```bash
python ai/run.py train --config ai/configs/mlp.yaml --fold 3
```

Uma região cinemática específica da grade 5×5:

```bash
python ai/run.py train --config ai/configs/mlp.yaml --et-bin 2 --eta-bin 0
```

### 2. Avaliação

```bash
python ai/run.py evaluate --config ai/configs/mlp.yaml [--et-bin 2 --eta-bin 0]
```

Recortar pontos de operação e gráficos sem reinferir (usa os scores já salvos):

```bash
python ai/run.py evaluate --config ai/configs/mlp.yaml --reuse-scores
```

### 3. Tabelão

```bash
python ai/run.py report --config ai/configs/mlp.yaml --formats tex,pdf
```

Comparando vários modelos numa tabela só (a ordem dos nomes é a ordem das linhas):

```bash
python ai/run.py report --config ai/configs/mlp.yaml --models MLP,CNN2D
```

Ou todos os que já foram avaliados:

```bash
python ai/run.py report --config ai/configs/mlp.yaml --all-models
```

Saída em `results/<MODEL>/tabelao/` para um modelo só, e em `results/comparison/tabelao/`
quando há mais de um.

> Para a comparação ser justa, os YAMLs dos modelos comparados precisam concordar em
> `data_path`, `max_files`, `test_size` e `seed` — é isso que garante que todos foram
> avaliados exatamente sobre as mesmas linhas de teste. O `report` compara as contagens de
> sinal/ruído do holdout de cada modelo por região e avisa se elas divergirem.

#### Como o `report` acha os modelos

Ele varre a árvore de resultados (`--results-root`, padrão `results/`) procurando
`artifacts/manifest.json` — o arquivo que o `train` escreve para cada região. O nome do modelo
sai de dentro do manifesto, não do nome da pasta. A ancoragem é no manifesto (e não nas
métricas) de propósito: assim uma região **treinada mas não avaliada** aparece no inventário e
é reportada como faltante, em vez de virar um buraco silencioso na tabela.

Para ver o inventário sem construir nada:

```bash
python ai/run.py report --all-models --list
```

```
🔎 Found 9 trained region(s) under 'results':
   CNN2D    et2_eta0    3/3 folds evaluated    results/CNN2D/et2_eta0
   MLP      et2_eta0    3/3 folds evaluated    results/MLP/et2_eta0
   MLP      et4_eta1    NOT EVALUATED          results/MLP/et4_eta1
```

Toda execução do `report` imprime esse inventário antes de montar a tabela e avisa, com o
comando exato para corrigir, quando uma região foi treinada e não avaliada, ou quando só parte
dos folds foi avaliada.

Com `--models` ou `--all-models`, o `--config` é dispensável — o `report` lê a árvore de
resultados, não os dados.

### 4. SLURM

```bash
./slurm_kfold.sh 5 ai/configs/mlp.yaml    # 1 job por fold + avaliação encadeada
```

```bash
./slurm_bins.sh ai/configs/mlp.yaml       # 1 job por bin da grade 5x5 + tabelão final
```

Ambos usam `--dependency=afterok`, então a avaliação e o tabelão só disparam quando os treinos terminam com sucesso.

---

## 📂 Artefatos gerados

```
results/<MODEL>[/et<i>_eta<j>]/
├── artifacts/
│   ├── manifest.json          # dataset, split, seed, hiperparâmetros
│   ├── preprocessor.joblib    # preprocessador ajustado SÓ no treino
│   └── test_indices.npy       # índices do holdout
├── checkpoints/
│   ├── fold_N.ckpt            # melhor checkpoint do fold (nome fixo)
│   └── fold_N.json            # pos_weight, melhor métrica, épocas, kwargs
├── history/fold_N.csv         # loss de treino/validação por época
├── scores/fold_N.parquet      # y_true, y_prob, cl_et, cl_eta do holdout
├── metrics/
│   ├── per_fold.csv           # métricas globais por fold
│   ├── operating_points.csv   # PD/SP/FA por (fold, ponto de operação)
│   └── folds_long.csv         # tabela canônica desta região
└── plots/                     # ROC, PR, matriz de confusão, loss, ROC dos folds

results/<MODEL>/tabelao/       # ou results/comparison/tabelao/ ao comparar modelos
├── tabelao_long.csv           # tabela canônica agregada (fonte da verdade)
├── tabelao_<ponto>.tex        # fragmento LaTeX para \input{}
└── tabelao_<ponto>.pdf        # render da tabela (sem precisar de LaTeX)
```

Todos os CSVs são **sobrescritos** a cada execução (antes eram anexados, o que fazia execuções antigas se acumularem como se fossem folds extras).

---

## 📊 O tabelão de validação cruzada

Tabela no formato ATLAS/Ringer: linhas são regiões de $|\eta|$, grupos de colunas são regiões de $E_T$, e cada grupo traz $P_D$ / $SP$ / $F_A$ como média ± desvio entre os folds. Quando mais de um modelo foi avaliado, cada região ganha **uma linha por modelo** (coluna `Model`), no lugar das linhas `Reference` / `Cross Validation` do formato original.

A fonte da verdade é o CSV **longo** (`tabelao_long.csv`): uma linha puramente numérica por `(model, et_bin, eta_bin, fold, operating_point)`. O `.tex` e a figura são derivados dele, então existe um único lugar onde os números são produzidos e vários onde são formatados.

O `.tex` gerado precisa dos pacotes:

```latex
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{graphicx}
```

**Ajuste ao ponto de operação.** Para cada $P_D$ alvo, o limiar é o quantil $(1 - P_D)$ da distribuição de scores de sinal — ou seja, toda rede é ajustada para entregar exatamente aquele $P_D$ (coluna destacada em verde). O que efetivamente distingue os modelos é o $SP$ e o $F_A$ resultantes, e é por isso que a comparação entre arquiteturas se lê direto na vertical dessas duas colunas.

**Convenção do desvio.** Todos os folds são avaliados no **mesmo** holdout estratificado. Logo o ± reportado é a variância *do modelo* entre folds, não a variância amostral do conjunto de teste.

**Sem linha de referência.** O dataset não traz coluna de decisão do T2Calo, então a tabela hoje só tem as linhas de `Cross Validation`, com alvos fixos de $P_D$ (90/95/99% por padrão). Uma linha de referência exigiria um arquivo externo de eficiências por região.

---

## ⚖️ Estratégia de Balanceamento de Classes (Weighted Loss)

O repositório utiliza **função de custo ponderada** (`nn.BCEWithLogitsLoss(pos_weight=pos_weight)`) em substituição ao *undersampling* aleatório, preservando 100% dos dados originais.

### 1. Cálculo Dinâmico de `pos_weight`

$$\text{pos\_weight} = \frac{N_{\text{negativos}}}{N_{\text{positivos}}}$$

- Calculado estritamente sobre as amostras de **treino** de cada fold (`train_ids`), sem vazamento de teste ou validação.
- Registrado como buffer (`register_buffer("pos_weight", ...)`), acompanhando o dispositivo sem ser otimizado.
- Salvo no sidecar `checkpoints/fold_N.json` — ele é excluído de `save_hyperparameters`, então precisa ser reinjetado ao recarregar o checkpoint.
- Partição estratificada com `seed` fixo garante que `CNN2D` e `MLP` comparem resultados sob os mesmos splits.

### 2. Métricas para dados desbalanceados

Precision/Recall/F1, AUC-ROC, AUC-PR e o **SP Index** (`sqrt(sqrt(pd*(1-fa)) * (pd+1-fa)/2)`), que é a métrica monitorada pelo Early Stopping e pelo ModelCheckpoint.

### 3. Vazamento de normalização

O `StandardScaler` é ajustado **apenas nas linhas de treino** (`PreprocessMLP.fit`) e persistido em `artifacts/preprocessor.joblib`, para que a avaliação reproduza exatamente a mesma transformação.

---

## 🧹 Limpeza do Ambiente

```bash
make clean
```
