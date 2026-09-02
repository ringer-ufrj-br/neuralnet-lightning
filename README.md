# neuralnet-lightning

Orquestrador e pipeline de treinamento de redes neurais (baseado em PyTorch / PyTorch Lightning) voltado para análise de dados do ATLAS (CERN).

---

## 📌 Visão Geral

O projeto automatiza o fluxo de carregamento de dados (ex: arquivos Parquet), pré-processamento, criação e treinamento de modelos de deep learning (`MLP`, `CNN2D`), validação cruzada por K-Fold e avaliação de desempenho — incluindo o **tabelão** de validação cruzada em LaTeX.

O fluxo é dividido em **três comandos independentes**:

| Comando | O que faz | O que produz |
|---|---|---|
| `train` | Treina os folds da validação cruzada | Checkpoints, preprocessador, índices de validação por fold, manifesto |
| `evaluate` | Reinferência dos folds sobre a região inteira | Scores, métricas por fold, gráficos, fatia do tabelão daquela região |
| `report` | Agrega todas as regiões avaliadas, de um ou vários modelos | Tabelão em `.tex`, figura da tabela e o CSV longo canônico |

A separação existe para que **re-avaliar não exija retreinar**: recortar pontos de operação, refazer gráficos ou remontar a tabela lê apenas artefatos em disco.

---

## 📁 Estrutura Principal do Projeto

- **`ai/`**: módulos de inteligência artificial.
  - `ai/run.py`: entrypoint com os subcomandos `train` / `evaluate` / `report`.
  - `ai/pipeline/base.py`: pipeline compartilhado (treino, avaliação, persistência de artefatos).
  - `ai/pipeline/registry.py`: mapeia o nome em `model:` do YAML para o pipeline correspondente.
  - `ai/pipeline/pipeline_*.py`: ligam um modelo ao seu preprocessador.
  - `ai/models/base.py`: `BaseBinaryClassifier`, a base de todas as arquiteturas (loss ponderada,
    métricas, índice SP, otimizador).
  - `ai/models/`: as arquiteturas em si.
  - `ai/preprocess/base.py`: `BasePreprocessor`, a base dos preprocessadores.
  - `ai/preprocess/`: preprocessadores de cada arquitetura.
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

### Em cluster (SLURM)

Faça o `make venv` **de dentro de uma alocação**, não no nó de login: a instalação do torch
baixa alguns GB e nós de login costumam limitar CPU/memória. Alocar uma GPU no mesmo
partition dos jobs também permite confirmar que o CUDA enxerga o dispositivo:

```bash
srun -p gpu --gres=gpu:1 --pty bash
```

```bash
make venv && ./neuralnet-env/bin/python -c "import torch; print(torch.cuda.is_available())"
```

O venv precisa ficar num sistema de arquivos que os nós de computação enxergam (o próprio
diretório do repositório, não um scratch local do nó).

Feito isso, **a submissão em si roda no nó de login** — `slurm_*.sh` só chama `sbatch`. Os
scripts resolvem o interpretador em `neuralnet-env/bin/python` por caminho absoluto e passam
`--chdir` para o repositório, então não dependem do ambiente do nó de login nem de onde você
submete. Para usar outro interpretador (módulo do cluster, conda, imagem):

```bash
PYTHON=/caminho/para/python ./slurm_bins.sh ai/configs/mlp.yaml
```

Os dados podem ser copiados para `data/` com `make copy-data`, ou — evitando duplicar alguns
GB — apontando `data_path` do YAML direto para o caminho compartilhado.

---

## ⚙️ Configuração (`ai/configs/*.yaml`)

```yaml
model: "MLP"                       # Modelo a ser utilizado (MLP | CNN2D | Fused)
data_path: data/parquet/           # Caminho para os dados
max_files: 100                     # Quantidade máxima de arquivos por pasta (omita para todos)
label_col: "label"                 # Coluna de rótulo
max_epochs: 5000                   # Teto de épocas; quem para o treino é o Early Stopping
batch_size: 1024                   # Tamanho do batch
learning_rate: 0.001               # Taxa de aprendizado
patience: 50                       # Épocas sem melhora de val_sp antes de parar
n_splits: 10                       # Folds da validação cruzada (1 = sem validação cruzada)
n_inits: 5                         # Inicializações independentes por fold; a melhor é mantida
threshold: 0.8                     # Limiar fixo das métricas globais
seed: 42                           # Semente da partição de folds
```

### Partição e o que a avaliação cobre

A partição estratificada de `n_splits` folds separa treino de validação **durante o treino**: é
ela que alimenta o Early Stopping e a escolha entre as inicializações. Não há holdout separado.

O `evaluate` roda cada fold sobre a **região inteira**, incluindo as linhas em que aquele fold
treinou. As eficiências relatadas, portanto, não são estritamente fora de amostra — é uma
escolha deliberada, para que os números cubram todo o espaço de fase.

Cada `scores/fold_N.parquet` traz a coluna booleana `in_sample`, marcando as linhas em que
aquele fold treinou. Quem quiser o recorte estritamente fora de amostra tem os dados à mão:

```python
d = pd.read_parquet("results/MLP/et2_eta0/scores/fold_1.parquet")
fora = d[~d.in_sample]
```

### Inicializações (`n_inits`)

Cada fold é treinado `n_inits` vezes a partir de pesos aleatórios diferentes (a partição dos
dados não muda), e fica apenas a inicialização com melhor `val_sp`. Serve para reduzir a
influência de mínimos locais. Os checkpoints perdedores são apagados, então o disco guarda um
checkpoint por fold independentemente de `n_inits`. Com `n_inits: 1` o comportamento é o de
sempre.

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

Todos os modelos já avaliados, sem precisar passar nada:

```bash
python ai/run.py report
```

`--config` e `--models` são **alternativas**, não complementos — cada um é uma forma de dizer
quais modelos entram na tabela:

```bash
python ai/run.py report --config ai/configs/mlp.yaml     # o modelo nomeado no YAML
```

```bash
python ai/run.py report --models MLP,CNN2D               # esses, nessa ordem de linhas
```

Saída em `results/<MODEL>/tabelao/` para um modelo só, e em `results/comparison/tabelao/`
quando há mais de um.

> Para a comparação ser justa, os YAMLs dos modelos comparados precisam concordar em
> `data_path`, `max_files`, `n_splits` e `seed` — é isso que garante que todos foram
> avaliados exatamente sobre as mesmas linhas de teste. O `report` compara as contagens de
> sinal/ruído das linhas avaliadas de cada modelo por região e avisa se elas divergirem.

#### Como o `report` acha os modelos

Ele varre a árvore de resultados (`--results-root`, padrão `results/`) procurando
`artifacts/manifest.json` — o arquivo que o `train` escreve para cada região. O nome do modelo
sai de dentro do manifesto, não do nome da pasta. A ancoragem é no manifesto (e não nas
métricas) de propósito: assim uma região **treinada mas não avaliada** aparece no inventário e
é reportada como faltante, em vez de virar um buraco silencioso na tabela.

Para ver o inventário sem construir nada:

```bash
python ai/run.py report --list
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

O `--config` é sempre dispensável: o `report` lê a árvore de resultados, não os dados.

#### Tabela integrada (arquivo separado)

Além de uma tabela por ponto de operação com a grade por região, o `report` salva a **tabela
integrada** em arquivos próprios: `tabelao_integrated.tex`, `.pdf`/`.png` e o
`tabelao_integrated_long.csv`. Ela pooleia todas as regiões e traz uma linha por modelo, com
um grupo de colunas por ponto de operação — o resultado inteiro numa linha por modelo.

A integração é ponderada pela população, não pela média das taxas: cada região tem seu próprio
limiar, então o número integrado é razão de contagens somadas.

$$P_D^{int} = \frac{\sum_r P_D^r \, N_{sinal}^r}{\sum_r N_{sinal}^r}, \qquad
  F_A^{int} = \frac{\sum_r F_A^r \, N_{ruído}^r}{\sum_r N_{ruído}^r}$$

O pooling acontece **por fold**, antes de agregar, então o ± da tabela integrada é a dispersão
real entre folds do número integrado. `threshold` e as AUCs ficam vazios (o limiar é por região
e as AUCs não se combinam a partir de sumários).

Para pular a tabela integrada:

```bash
python ai/run.py report --no-integrated
```

### 4. SLURM

A grade 5×5 vai como **um job array** de 25 tarefas — cada tarefa deriva seu par (et, eta) do
`SLURM_ARRAY_TASK_ID`, treina os folds daquele bin e avalia em seguida. Ao fim do array, um job
dependente monta o tabelão:

```bash
./slurm_bins.sh ai/configs/mlp.yaml
```

Segundo argumento opcional limita quantas tarefas rodam em paralelo (`--array=0-24%N`), para
não tomar todas as GPUs da fila:

```bash
./slurm_bins.sh ai/configs/mlp.yaml 4
```

Cancelar a grade inteira é `scancel <id-do-array>`; uma região só é `scancel <id>_<indice>`.

O tabelão encadeia com `--dependency=afterok`, então só dispara quando todas as tarefas do
array terminam com sucesso.

---

## 🧩 Adicionando uma arquitetura

Uma arquitetura nova são **três arquivos curtos**. Todo o resto — k-fold, binning
cinemático, batching, métricas, índice SP, EarlyStopping, checkpoints, scoring, gráficos,
tabelão e a grade SLURM — já vem pronto e funciona igual para qualquer modelo.

O exemplo abaixo cria uma rede chamada `MinhaRede`.

### 1. O modelo — `ai/models/minha_rede.py`

Herde de `BaseBinaryClassifier` e escreva apenas `build_network`, devolvendo as camadas:

```python
import torch.nn as nn
from ai.models.base import BaseBinaryClassifier


class ModelMinhaRede(BaseBinaryClassifier):
    def build_network(self, input_dim: int = 100, hidden: int = 16) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
```

Não escreva `__init__` — a base monta a loss ponderada, as métricas, o índice SP e o otimizador.
Cada argumento de `build_network` vira hiperparâmetro salvo, disponível como `self.hparams.hidden`
e restaurado do checkpoint. A rede devolve **logits crus**: a loss aplica o sigmoid.

### 2. O preprocessador — `ai/preprocess/minha_rede.py`

Herde de `BasePreprocessor` e escreva `required_columns` (quais colunas ler do parquet, entre as
300+ disponíveis) e `transform` (DataFrame → array float32), terminando com `self.normalize(...)`:

```python
class PreprocessMinhaRede(BasePreprocessor):
    def required_columns(self, available):
        return [c for c in available if c.startswith("cl_ring_")]

    def transform(self, df):
        X = df[self.required_columns(list(df.columns))].to_numpy(dtype=np.float32)
        return self.normalize(X)
```

`save`, `load`, `fit_transform` e `get_labels` já vêm da base. Escreva `fit` apenas se houver
estado a aprender dos dados (um scaler, uma média); o `fit` padrão é no-op, que é o certo para
um preprocessador sem estado como o da MLP.

### 3. O pipeline — `ai/pipeline/pipeline_minha_rede.py`

O nome do arquivo **precisa começar com `pipeline_`**: é assim que o registro o encontra.

```python
@register_pipeline("MinhaRede")     # o valor que vai em `model:` no YAML
class PipelineMinhaRede(BasePipeline):
    model_class = ModelMinhaRede
    preprocessor_class = PreprocessMinhaRede
```

Se a arquitetura precisa de um valor que só se conhece depois do preprocessamento — tipicamente
a dimensão de entrada — acrescente:

```python
    def build_model_kwargs(self, X):
        return {"input_dim": int(X.shape[1])}
```

### 4. Rodar

Aponte `model: "MinhaRede"` no YAML e use os mesmos comandos de sempre:

```bash
python ai/run.py train    --config ai/configs/minha_rede.yaml
python ai/run.py evaluate --config ai/configs/minha_rede.yaml
python ai/run.py report   --config ai/configs/minha_rede.yaml
```

Para conferir que a arquitetura foi reconhecida:

```bash
python -c "from ai.pipeline.registry import available_pipelines; print(available_pipelines())"
```

### Ganchos opcionais

Sobrescreva apenas se precisar; nenhum é obrigatório:

| Gancho | Quando usar |
|---|---|
| `forward` | a rede não é um único módulo chamável (ex.: dois ramos — veja `ai/models/fused.py`) |
| `compute_loss` | losses auxiliares, além da principal |
| `build_metrics` | acrescentar ou remover métricas |
| `configure_optimizers` | outro otimizador ou um scheduler |
| `build_preprocessor` | o preprocessador precisa de argumentos de construção |

---

## 📂 Artefatos gerados

```
results/<MODEL>[/et<i>_eta<j>]/
├── artifacts/
│   ├── manifest.json          # dataset, split, seed, hiperparâmetros
│   ├── preprocessor.joblib    # preprocessador ajustado SÓ no treino
│   └── val_indices_fold_N.npy # linhas que o fold N validou (fora de amostra)
├── checkpoints/
│   ├── fold_N.ckpt            # melhor checkpoint do fold (nome fixo)
│   └── fold_N.json            # pos_weight, melhor métrica, épocas, kwargs
├── history/fold_N.csv         # loss de treino/validação por época
├── scores/fold_N.parquet      # y_true, y_prob, cl_et, cl_eta, in_sample (região inteira)
├── metrics/
│   ├── per_fold.csv           # métricas globais por fold
│   ├── operating_points.csv   # PD/SP/FA por (fold, ponto de operação)
│   └── folds_long.csv         # tabela canônica desta região
└── plots/                     # ROC, PR, matriz de confusão, loss, ROC dos folds

results/<MODEL>/tabelao/               # ou results/comparison/tabelao/ ao comparar modelos
├── tabelao_long.csv                   # tabela canônica agregada (fonte da verdade)
├── tabelao_<ponto>.tex                # por região: fragmento LaTeX para \input{}
├── tabelao_<ponto>.pdf                # por região: render sem precisar de LaTeX
├── tabelao_integrated_long.csv        # integrado: números poolados por fold
├── tabelao_integrated.tex             # integrado: fragmento LaTeX
└── tabelao_integrated.pdf             # integrado: render
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

**Convenção do desvio.** Todos os folds são avaliados sobre o **mesmo** conjunto de linhas (a região inteira). Logo o ± reportado é a variância *do modelo* entre folds, não a variância amostral do conjunto avaliado.

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

### 3. Normalização da entrada

Cada evento é normalizado pela soma absoluta das suas próprias features,
$r^{\prime}_i = r_i / |\sum_j r_j|$, de modo que a rede enxerga apenas o **formato** da
deposição de energia e não a escala absoluta — que já é tratada pelo binning em $E_T$.

Isso vive no preprocessador (`BasePreprocessor.normalize`, chamado no fim de cada `transform`),
e não no modelo: é uma propriedade da representação de entrada, não da arquitetura. Também é
mais barato — a normalização é calculada uma vez por conjunto, e não a cada batelada de cada
época. O array persistido é exatamente o que a rede enxerga.

---

## 🧹 Limpeza do Ambiente

```bash
make clean
```
