# neuralnet-lightning

Framework para treinar redes neurais (PyTorch Lightning) com os dados do Ringer do ATLAS:
classificação elétron × jato com validação cruzada, uma rede para cada região de
$E_T$ × $|\eta|$, métricas, gráficos e a tabela final de eficiências (o "tabelão").

Cada experimento é um arquivo YAML que junta um modelo e um dataset. Um dataset novo é um YAML,
e um modelo novo é um arquivo Python.

> **Resultados antigos:** resultados gerados antes da reestruturação (em `results/<MODELO>/...`,
> com `fold_N.json`) não são lidos pelo código novo e precisam ser refeitos.

---

## 📌 Como funciona

| Comando | O que faz |
|---|---|
| `train` | Treina a rede com validação cruzada e salva todos os modelos (cada fold × inicialização) |
| `evaluate` | Escolhe a melhor inicialização de cada fold, passa nos dados da região e gera métricas e gráficos |
| `report` | Junta todas as regiões no tabelão (LaTeX e HTML) |
| `grid` | Lista as regiões da grade, para scripts que distribuem o trabalho |

Cada passo lê o que o anterior salvou em disco. Assim dá para refazer a avaliação ou o tabelão
sem treinar de novo.

---

## ⚙️ Instalação

```bash
make venv                            # cria o neuralnet-env e instala o pacote (pip install -e .[test])
source neuralnet-env/bin/activate
make test                            # ~10 s em CPU, sobre dados sintéticos
```

As versões estão fixadas no `pyproject.toml`. O `requirements.txt` só existe para apontar o
índice do `torch==2.13.0+cu126`: os nós de GPU usam driver 550 (CUDA 12.4), e o `torch` padrão
do PyPI exige driver 580+.

**No cluster**, rode o `make venv` de dentro de uma alocação, e não no nó de login (a
instalação do torch é pesada). Por exemplo:

```bash
srun -p gpu --pty bash
make venv && ./neuralnet-env/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Deixe o ambiente dentro da pasta do repositório, para os nós de computação enxergarem.

**Os dados** são lidos direto da pasta compartilhada (os caminhos estão em
`configs/datasets/`). Quem quiser uma cópia local pode usar `make copy-data`
(`DATA_SRC=/outro/caminho` para copiar outra pasta) e apontar o `data_path` para ela.

---

## 🧠 Modelos

| `model:` | O que é |
|---|---|
| `MLP` | Uma camada escondida (`hidden`, default 5) sobre metade dos anéis de cada camada (50), com norm1 (cada evento dividido pela soma dos seus anéis) |
| `CNN2D` | Rede convolucional sobre as imagens de células do calorímetro (uma camada = um canal) |
| `Fused` | Anéis e imagens de células em dois ramos, que se juntam no final |

`CNN2D` e `Fused` precisam das imagens de células (`cl_cells_*`), que só o mc25 tem. Num
dataset sem elas, o `train` para logo no carregamento com
`❌ Column(s) ['cl_cells_presampler', ...] are not in the dataset`.

---

## 📝 Configuração

Um experimento:

```yaml
name: mlp_mc21                # o experimento: resultados em <results_root>/<name>/<região>/
model: MLP                    # a arquitetura (tabela acima)
results_root: results         # opcional

dataset: datasets/mc21.yaml   # um arquivo de configs/datasets/ (relativo a este config)

model_args:                   # opcional: sobrescreve os defaults do build_network do modelo
  hidden: 5

training:                     # tudo opcional; os valores abaixo são os defaults
  max_epochs: 5000
  batch_size: 1024
  learning_rate: 0.001
  patience: 50                # épocas sem melhorar o SP antes de parar
  n_splits: 10                # folds (1 = um modelo só, com 20% de validação)
  n_inits: 5                  # inicializações por fold; o evaluate usa a melhor
  seed: 42                    # semente da divisão em folds
  accelerator: auto           # auto, cpu, gpu
  devices: auto

evaluation:
  operating_points: {tight: 0.90, medium: 0.95, loose: 0.99}   # nome -> P_D alvo
```

Um dataset (`configs/datasets/mc21.yaml`):

```yaml
data_path: /mnt/shared/.../mc21_isabela_qt_2sigma_restriction/electron_ringer.parquet
rings_col: "TrigEMClusterContainer.ringsE"   # uma coluna de lista com os 100 anéis,
                                             # ou "cl_ring_%i": uma coluna por anel
et_col: "TrigEMClusterContainer.et"          # em MeV
eta_col: "TrigEMClusterContainer.eta"
label_col: target             # sem isso, o label vem do nome do arquivo (Zee = 1, JF17 = 0)
max_files: 10                 # opcional: só os N primeiros arquivos de cada pasta
```

O `dataset:` de um experimento aceita três formas:

```yaml
dataset: datasets/mc25.yaml                    # o arquivo
dataset: {file: datasets/mc25.yaml, max_files: 100}   # o arquivo, sobrescrevendo algumas chaves
dataset: {data_path: ..., rings_col: ..., et_col: ..., eta_col: ...}   # tudo inline
```

O caminho de `file` é relativo à pasta do config que o cita, então o comando pode rodar de
qualquer diretório. `data_path`, `rings_col`, `et_col` e `eta_col` são obrigatórios: não há
colunas default escondidas no código.

O config é validado na carga: uma chave desconhecida, como `bacth_size`, ou um valor do tipo
errado gera erro em vez de ser ignorado. Os defaults ficam só nas dataclasses de
`neuralnet_lightning/config.py`.

`name` e `model` são independentes. O mesmo modelo em outro dataset, ou com outros
`model_args`, é só outro config com outro `name`, sem classe nova. É assim que `mlp.yaml`
(`mlp_mc25`) e `mlp_mc21.yaml` (`mlp_mc21`) usam a mesma `MLP`.

| Config | `name` | Modelo | Dataset |
|---|---|---|---|
| `configs/mlp.yaml` | `mlp_mc25` | `MLP` | mc25 |
| `configs/mlp_mc21.yaml` | `mlp_mc21` | `MLP` | mc21 |
| `configs/cnn2d.yaml` | `cnn2d_mc25` | `CNN2D` | mc25, 100 arquivos por pasta |
| `configs/fused.yaml` | `fused_mc25` | `Fused` | mc25, 100 arquivos por pasta |

---

## 🗃️ Adicionando um dataset

Um dataset novo é um arquivo em `configs/datasets/`. Para o ⟨μ⟩ = 200, por exemplo:

1. Olhe as colunas de um arquivo:

   ```python
   import polars as pl
   print(pl.read_parquet_schema("/caminho/do/mu200/00000000.parquet"))
   ```

2. Crie `configs/datasets/mu200.yaml` com o que achou:

   ```yaml
   data_path: /caminho/do/mu200            # pasta, arquivo ou glob; subpastas são lidas
   rings_col: "TrigEMClusterContainer.ringsE"
   et_col: "TrigEMClusterContainer.et"     # em MeV (a grade de E_T está em MeV)
   eta_col: "TrigEMClusterContainer.eta"
   label_col: target                       # ou omita, se o label vier do nome do arquivo
   ```

3. Aponte um experimento para ele, com um `name` novo:

   ```yaml
   # configs/mlp_mu200.yaml
   name: mlp_mu200
   model: MLP
   dataset: datasets/mu200.yaml
   training: {...}                          # copie de mlp_mc21.yaml para comparar
   ```

4. Teste rápido numa região, antes da grade:

   ```bash
   python -m neuralnet_lightning train --config configs/mlp_mu200.yaml --et-bin 2 --eta-bin 0
   ```

O `make test` verifica que todo arquivo de `configs/` e `configs/datasets/` é válido.

Se o label vier do nome do arquivo com outros padrões que não Zee/JF17, ajuste `PATTERNS` em
`neuralnet_lightning/data/labels.py`. Um arquivo que não casa com nenhum padrão é recusado, em
vez de virar um label errado.

---

## 🚀 Rodando

Para uma região:

```bash
python -m neuralnet_lightning train    --config configs/mlp.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning evaluate --config configs/mlp.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning report   --config configs/mlp.yaml
```

Sem `--et-bin`/`--eta-bin`, uma rede só cobre o espaço de fase inteiro (`<name>/all/`).

Opções úteis:

- `train --fold N --init M` treina um único modelo, como cada tarefa do SLURM faz.
- `train --accelerator cpu --devices 1` sobrescreve o config.
- `evaluate --reuse-scores` refaz métricas e gráficos sem passar a rede de novo.
- `evaluate --no-plots` calcula só as métricas.
- `report --names cnn2d_mc25,fused_mc25` compara experimentos, nessa ordem de linhas.
- `report --list` mostra o que já foi treinado e avaliado, e o comando que falta para cada buraco.
- `report --no-integrated` pula a tabela integrada.

### A grade inteira na sua máquina

```bash
./scripts/run_local_grid.sh configs/mlp.yaml
```

Treina e avalia as 25 regiões, uma de cada vez, e monta o tabelão. As regiões já avaliadas são
puladas (`FORCE=1` refaz tudo).

### A grade inteira no cluster (SLURM)

```bash
./scripts/slurm_bins.sh configs/mlp.yaml       # a grade inteira
./scripts/slurm_bins.sh configs/mlp.yaml 40    # no máximo 40 tarefas ao mesmo tempo
```

Rode do nó de login. O script valida o config (um erro de digitação falha na hora, e não em 625
tarefas) e submete três etapas, cada uma esperando a anterior terminar sem erro:

1. um job array com um treino por tarefa (região × fold × inicialização);
2. um job array com uma avaliação por região, usando a melhor inicialização de cada fold;
3. um job que monta o tabelão.

Variáveis: `PARTITION` (default `gpu`), `GRES`, `PYTHON` (default `neuralnet-env/bin/python`).
Os logs ficam em `<results_root>/<name>/slurm_logs/`. Para cancelar, `scancel <id>`.
**Para as MLPs, que rodam melhor em CPU, leia [scripts/README.md](scripts/README.md)**: sem
`SBATCH_MEM_PER_NODE` e `OMP_NUM_THREADS=1`, a grade ocupa um nó inteiro por tarefa.

---

## 🧩 Adicionando um modelo

Um modelo novo é **um arquivo** em `neuralnet_lightning/models/`. Ele é registrado sozinho,
sem lista para editar:

```python
# neuralnet_lightning/models/minha_rede.py
import torch.nn as nn

from neuralnet_lightning.binary.model import BaseBinaryClassifier
from neuralnet_lightning.models.registry import register_model
from neuralnet_lightning.preprocess.mlp import PreprocessMLP


@register_model("MinhaRede")                 # o nome que vai em `model:`
class MinhaRede(BaseBinaryClassifier):
    preprocessor = PreprocessMLP             # qual entrada a rede recebe

    def build_network(self, input_dim: int = 50, hidden: int = 16, dropout: float = 0.1) -> nn.Module:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, 1))
```

- Não escreva `__init__`. A base já cuida da loss (BCE com `pos_weight`), das métricas, do
  `val_sp` que o early stopping acompanha e do otimizador.
- Os argumentos de `build_network` são os hiperparâmetros da arquitetura. Eles ficam salvos no
  checkpoint e podem ser trocados pelo `model_args` do config. O que só se sabe depois do
  preprocessamento (como o `input_dim`) vem do `model_kwargs` do preprocessor.
- A rede devolve logits crus `(Batch, 1)`, sem sigmoid.
- Se precisar, sobrescreva `forward` (rede com mais de um ramo, veja
  `neuralnet_lightning/models/fused.py`), `compute_loss` (losses auxiliares), `build_metrics`
  ou `configure_optimizers`.

Se a entrada for outra, crie também um preprocessor em `neuralnet_lightning/preprocess/`. Por
exemplo, anéis + shower shapes do mc21:

```python
# neuralnet_lightning/preprocess/rings_ss.py
import numpy as np
import polars as pl

from neuralnet_lightning.preprocess.base import BasePreprocessor
from neuralnet_lightning.preprocess.mlp import selected_rings

SHOWER_SHAPES = ["HLTElectronContainer.reta", "HLTElectronContainer.eratio", "HLTElectronContainer.rhad"]


class PreprocessRingsSS(BasePreprocessor):
    def required_columns(self):
        return selected_rings() + SHOWER_SHAPES            # só essas colunas são lidas do disco

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        rings = self.normalize(self.extract(df, selected_rings()))   # norm1 só nos anéis
        return np.hstack([rings, self.extract(df, SHOWER_SHAPES)])

    def model_kwargs(self, X):
        return {"input_dim": X.shape[1]}
```

- O `df` é do polars: não converta para pandas, isso copiaria o dataframe inteiro.
- Se o preprocessor precisa aprender algo dos dados (média, desvio de cada shower shape),
  escreva um `fit`. O objeto ajustado é salvo com o treino e reaproveitado no `evaluate`. Atenção:
  o `fit` vê a região inteira, antes da divisão em folds.
- Uma normalização diferente é um preprocessor diferente (uma subclasse que sobrescreve
  `normalize`), para que cada checkpoint diga com qual normalização foi treinado.

**Shower shapes e tracking** não têm nomes canônicos: as colunas que não são anéis, Et, eta,
label ou id aparecem com o nome original do parquet, e mudam de dataset para dataset
(`cl_rhad` no mc25, `HLTElectronContainer.rhad` no mc21). O preprocessor acima só serve,
portanto, para datasets com esses nomes. Quando houver modelos usando essas variáveis em mais de
um dataset, o lugar de dar a elas nomes canônicos é o `DatasetSchema`
(`neuralnet_lightning/data/schema.py`), como já é feito com os anéis: um campo novo (por
exemplo, `shower_shape_cols`) vira uma chave do bloco `dataset:` sozinho.

Para ver os modelos registrados:

```bash
python -c "from neuralnet_lightning.models.registry import available_models; print(available_models())"
```

---

## ✂️ Adicionando uma modificação do dataset

O que vale para **qualquer** modelo (um corte de qualidade, uma coluna derivada, um alvo novo)
fica na camada de dados, em `neuralnet_lightning/data/loader.py`, como expressão polars aplicada
ao `LazyFrame` antes do `collect`. Assim ela roda dentro da leitura em streaming, e só as linhas
e colunas necessárias chegam à memória. Por exemplo, um corte:

```python
# em load_dataframe, logo depois de `frame = schema.scan(files)`:
frame = frame.filter(pl.col("TrigEMClusterContainer.isGoodRinger"))
```

Antes da projeção, o filtro enxerga as colunas com o nome original do parquet, e o polars lê a
coluna do corte sem que ela vá parar no DataFrame final. Para tornar a modificação configurável
por dataset, acrescente um campo ao `DatasetSchema` em `neuralnet_lightning/data/schema.py`:
ele vira uma chave do bloco `dataset:` automaticamente.

O que é específico de **uma** arquitetura (normalização, seleção de features) fica no
preprocessor dela, em `neuralnet_lightning/preprocess/`.

---

## 📂 O que fica salvo

```
results/<name>/<região>/          região = et<i>_eta<j>, ou "all" sem binning
├── artifacts/     o preprocessador ajustado e os eventos de validação de cada fold
├── checkpoints/   fold_N_init_M.ckpt e o .json com os detalhes do treino
├── history/       a loss de cada época, por fold e inicialização
├── logs/          logs do Lightning
├── scores/        a nota da rede para cada evento da região, por fold (+ et, eta, in_sample)
├── metrics/       folds_long.csv: P_D, SP e F_A por fold e ponto de operação
└── plots/         ROC, PR, matriz de confusão, curvas de loss, ROC por fold

results/<name>/pd_table/          (ou results/comparison/pd_table/ ao comparar experimentos)
├── pd_table_<ponto>.tex e .html       o tabelão de cada ponto de operação
├── pd_table_integrated.tex e .html    uma linha por experimento, juntando todas as regiões
└── pd_table_long.csv                  os números por trás das tabelas

results/<name>/slurm_logs/        logs das tarefas do SLURM
```

---

## 📊 Como ler o tabelão

- As linhas são as regiões de $|\eta|$ e as colunas, as de $E_T$. Cada célula traz $P_D$,
  $SP$ e $F_A$, com média e desvio entre os folds.
- Cada rede é ajustada para acertar o $P_D$ alvo (coluna em verde). O que diferencia os
  modelos é o $SP$ e o $F_A$.
- A tabela integrada junta todas as regiões, cada uma pesando pelo seu número de eventos.
- Para comparar experimentos de forma justa, os configs precisam usar os mesmos dados,
  `max_files`, `n_splits` e `seed`. O `report` avisa quando isso não acontece.
- O `.tex` precisa de `\usepackage{booktabs}`, `\usepackage[table]{xcolor}` e
  `\usepackage{graphicx}`.

---

## 🗺️ Organização do código

```
neuralnet_lightning/
├── __main__.py      CLI
├── config.py        dataclasses do YAML (os únicos defaults) e o `dataset:` em arquivo
├── binning.py       grade Et × |eta| e Region
├── paths.py         RunPaths: onde fica cada artefato de uma (experimento, região)
├── pipeline.py      orquestrador: config → loader → preprocessor → treino / avaliação
├── training.py      K-Fold, TensorBatchLoader, dataset inteiro na GPU quando cabe
├── evaluation.py    melhor inicialização por fold, inferência, scores/*.parquet
├── data/            schema (vocabulário canônico), labels pelo nome do arquivo, loader
├── preprocess/      DataFrame canônico → array do modelo (mlp, cnn2d, fused)
├── models/          registry + um arquivo por arquitetura (mlp, cnn2d, fused)
└── binary/          tudo que é da tarefa elétron × jato: modelo base, SP, pontos de
                     operação, gráficos, tabelão
```

O núcleo (dados, config, registry, treino, avaliação, CLI) não sabe que a tarefa é binária. Ele
lê da classe do modelo o que precisa: o preprocessor, a métrica monitorada e a direção dela, os
kwargs por fold (o `pos_weight`), o que estratificar e como resumir os scores. Esse contrato está
descrito em `neuralnet_lightning/models/registry.py`.

---

## 💡 Bom saber

- **Classes desbalanceadas:** em vez de descartar dados, a loss usa `pos_weight` = ruído/sinal,
  calculado só sobre os dados de treino de cada fold.
- **Quando o treino para:** o early stopping acompanha o $SP$ máximo de validação, varrendo
  todos os cortes possíveis a cada época.
- **A avaliação usa a região inteira**, inclusive os eventos com que cada fold treinou. Para
  ficar só com os eventos que o fold não viu, use a coluna `in_sample`:

  ```python
  import polars as pl
  d = pl.read_parquet("results/mlp_mc21/et2_eta0/scores/fold_1.parquet")
  fora = d.filter(~pl.col("in_sample"))
  ```

- **Reprodutível:** a ordem das linhas é determinística (pastas e arquivos ordenados), os folds
  dependem só de `seed` e cada (fold, init) tem a sua semente. Treinar de novo um único
  (fold, init) dá o mesmo modelo.
- **O `evaluate` recusa rodar** se o número de eventos da região mudou desde o treino (outro
  `max_files`, arquivos novos na pasta): os índices dos folds não valeriam mais.
- **Os dados no notebook:**

  ```python
  from neuralnet_lightning.config import load_config
  from neuralnet_lightning.data.loader import load_dataframe
  from neuralnet_lightning.binning import Region
  cfg = load_config("configs/mlp_mc21.yaml")
  df = load_dataframe(cfg.dataset, ["ring_0", "ring_1"], Region(2, 0))
  ```

---

## 🧹 Limpeza

```bash
make clean
```
