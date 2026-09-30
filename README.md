# neuralnet-lightning

Framework para treinar redes neurais (PyTorch Lightning) com os dados do Ringer do ATLAS.

Você escolhe um modelo e um dataset num arquivo de configuração, e o resto já vem pronto:
validação cruzada, uma rede para cada região de $E_T$ × $|\eta|$, métricas, gráficos e a
tabela final de eficiências (o "tabelão").

---

## 📌 Como funciona

São três comandos, um depois do outro:

| Comando | O que faz |
|---|---|
| `train` | Treina a rede com validação cruzada e salva todos os modelos (cada fold × inicialização) |
| `evaluate` | Escolhe a melhor inicialização de cada fold, passa nos dados e calcula métricas e gráficos |
| `report` | Junta todas as regiões no tabelão (LaTeX e HTML) |

Cada passo lê o que o anterior salvou em disco. Assim dá para refazer a avaliação ou a tabela
sem precisar treinar de novo.

---

## ⚙️ Instalação

```bash
make venv                            # cria o ambiente e instala as dependências
source neuralnet-env/bin/activate    # ativa o ambiente
```

**No cluster**, rode o `make venv` de dentro de uma alocação com GPU, e não no nó de login
(a instalação do torch é pesada). Por exemplo:

```bash
srun -p gpu --gres=gpu:1 --pty bash
make venv && ./neuralnet-env/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Deixe o ambiente dentro da pasta do repositório, para os nós de computação enxergarem.

**Os dados** podem ser copiados para `data/` com `make copy-data`, ou você pode apontar o
`data_path` do config direto para a pasta compartilhada.

---

## 🧠 Modelos disponíveis

| `model:` | O que é |
|---|---|
| `MLP` | Rede pequena (uma camada de 5 neurônios) sobre metade dos anéis. Normalização: norm1 (cada evento dividido pela soma dos seus anéis) |
| `MLP_MC21` | A MLP com um nome próprio, para os resultados do mc21 não se misturarem com os do mc25 |
| `CNN2D` | Rede convolucional sobre as imagens de células do calorímetro (uma camada = um canal) |
| `Fused` | Anéis e imagens de células em dois ramos, que se juntam no final |

---

## 📝 Configuração

Cada experimento é um arquivo YAML em `ai/configs/`:

```yaml
model: "MLP"          # qual modelo (tabela acima)
max_epochs: 5000      # limite de épocas; o early stopping costuma parar bem antes
batch_size: 1024
learning_rate: 0.001
patience: 50          # quantas épocas sem melhorar antes de parar
n_splits: 10          # número de folds da validação cruzada (1 = treina um modelo só)
n_inits: 5            # quantas vezes treinar cada fold com pesos iniciais diferentes; o evaluate usa a melhor
seed: 42              # semente da divisão em folds
```

Os pontos de operação do tabelão são tight (90%), medium (95%) e loose (99%) de $P_D$. Para
usar outros, acrescente:

```yaml
operating_points:
  tight: 0.90
  veryloose: 0.995
```

### Usando outro dataset

Sem nada declarado, o código espera o layout do mc25. Para outro dataset, informe no bloco
`dataset:` só o que muda:

```yaml
dataset:
  data_path: ../data/mc21_isabela_qt_2sigma_restriction/electron_ringer.parquet
  max_files: 100                              # opcional: só os N primeiros arquivos de cada pasta
  rings_col: "TrigEMClusterContainer.ringsE"  # uma coluna com os 100 anéis
                                              # (ou "cl_ring_%i": uma coluna por anel)
  et_col: "TrigEMClusterContainer.et"         # em MeV
  eta_col: "TrigEMClusterContainer.eta"
  label_col: target                           # sem isso, o rótulo vem do nome do arquivo
                                              # (Zee = sinal, JF17 = ruído)
results_root: results/mc21                    # cada dataset na sua pasta de resultados
```

`ai/configs/mlp_mc21.yaml` é um exemplo completo.

---

## 🚀 Rodando

A grade $E_T$ × $|\eta|$ tem 5 × 5 regiões, e cada região ganha a sua própria rede. Para uma
região:

```bash
python ai/run.py train    --config ai/configs/mlp.yaml --et-bin 2 --eta-bin 0
python ai/run.py evaluate --config ai/configs/mlp.yaml --et-bin 2 --eta-bin 0
python ai/run.py report   --config ai/configs/mlp.yaml
```

Sem `--et-bin`/`--eta-bin`, o treino usa todos os dados numa rede só.

Opções úteis:

- `evaluate --reuse-scores` refaz métricas e gráficos sem rodar a rede de novo.
- `evaluate --no-plots` calcula só as métricas.
- `report --models MLP,CNN2D` compara modelos, nessa ordem de linhas.
- `report` sem `--config` nem `--models` inclui todos os modelos que encontrar.
- `report --list` mostra o que já foi treinado e avaliado, sem montar a tabela.
- `report --no-integrated` pula a tabela integrada.

### A grade inteira na sua máquina

```bash
./scripts/run_local_grid.sh ai/configs/mlp.yaml
```

Treina e avalia as 25 regiões, uma de cada vez, e monta o tabelão no final. Se parar no meio,
é só rodar de novo: as regiões já avaliadas são puladas (`FORCE=1` refaz tudo).

### A grade inteira no cluster (SLURM)

```bash
./scripts/slurm_bins.sh ai/configs/mlp.yaml       # a grade inteira
./scripts/slurm_bins.sh ai/configs/mlp.yaml 4     # no máximo 4 tarefas rodando ao mesmo tempo
```

Rode do nó de login. O script faz três etapas, cada uma esperando a anterior terminar sem erro:

1. um job para cada treino (região × fold × inicialização);
2. um job por região, que avalia usando a melhor inicialização de cada fold;
3. um job que monta o tabelão.

O script já usa o Python do `neuralnet-env`; para usar outro, rode
`PYTHON=/caminho/para/python ./scripts/slurm_bins.sh ...`. Para cancelar, `scancel <id>`.

---

## 🧩 Adicionando uma rede nova

Uma rede nova são três arquivos curtos. Todo o resto (validação cruzada, grade, métricas,
gráficos, tabelão e SLURM) funciona sem mexer em mais nada. No exemplo, a rede se chama
`MinhaRede`.

**1. O modelo:** `ai/models/minha_rede.py`. Escreva só o `build_network`, que devolve as
camadas:

```python
import torch.nn as nn
from ai.models.base import BaseBinaryClassifier


class ModelMinhaRede(BaseBinaryClassifier):
    def build_network(self, input_dim: int = 100, hidden: int = 16) -> nn.Module:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
```

Não precisa de `__init__`: a base já cuida da loss, das métricas e do otimizador. Os
argumentos de `build_network` ficam salvos junto com o checkpoint. A rede devolve a saída crua,
sem sigmoid.

**2. O preprocessador:** `ai/preprocess/minha_rede.py`. Diz quais colunas ler e como elas
viram a entrada da rede:

```python
from ai.preprocess.base import BasePreprocessor


class PreprocessMinhaRede(BasePreprocessor):
    def required_columns(self, available):
        return [c for c in available if c.startswith("ring_")]

    def transform(self, df):        # df é um DataFrame do polars
        X = self.extract(df, self.required_columns(df.columns))  # matriz float32, sem NaN nem -999
        return self.normalize(X)    # norm1: cada evento dividido pela soma das suas features
```

O `df` é do polars, não do pandas: converter para pandas copiaria o dataframe inteiro. Monte a
matriz com `extract` (ou preenchendo um array já alocado) e trabalhe nela no lugar.

Se for só uma lista de colunas com norm1, basta declarar `feature_columns`: o `transform`
padrão extrai essas colunas e aplica o norm1 (é assim que a MLP faz). Se o preprocessador
precisa aprender algo dos dados, como uma média ou um scaler, escreva também um `fit`. Atenção:
o `fit` recebe a região inteira, antes da divisão em folds, então o que ele aprende inclui os
eventos de validação de cada fold.

**3. O pipeline:** `ai/pipeline/pipeline_minha_rede.py`. O nome do arquivo precisa começar com
`pipeline_`. É ele que liga o modelo ao preprocessador e dá o nome usado no config:

```python
from ai.pipeline.base import BasePipeline
from ai.pipeline.registry import register_pipeline
from ai.models.minha_rede import ModelMinhaRede
from ai.preprocess.minha_rede import PreprocessMinhaRede


@register_pipeline("MinhaRede")     # o nome que vai em `model:`
class PipelineMinhaRede(BasePipeline):
    model_class = ModelMinhaRede
    preprocessor_class = PreprocessMinhaRede

    # Opcional: valores que só se sabe depois do preprocessamento, como o tamanho da entrada.
    def build_model_kwargs(self, X):
        return {"input_dim": int(X.shape[1])}
```

**4. Rodar:** coloque `model: "MinhaRede"` num config e use os comandos de sempre. Para ver os
modelos registrados:

```bash
python -c "from ai.pipeline.registry import available_pipelines; print(available_pipelines())"
```

Mais algumas dicas:

- Para mudar só a normalização, crie um preprocessador novo e registre um pipeline com outro
  nome. Assim cada resultado diz com que normalização foi treinado.
- Se precisar, dá para sobrescrever também `forward` (rede com mais de um ramo, veja
  `ai/models/fused.py`), `compute_loss` (losses extras), `build_metrics` ou
  `configure_optimizers`.

---

## 📂 O que fica salvo

```
results/<MODELO>/et<i>_eta<j>/
├── artifacts/     o preprocessador ajustado e quais eventos cada fold usou na validação
├── checkpoints/   fold_N_init_M.ckpt (cada fold × inicialização) e o .json com os detalhes do treino
├── history/       a loss de cada época, por fold e inicialização
├── scores/        a nota que a rede deu a cada evento da região, por fold
├── metrics/       folds_long.csv: P_D, SP e F_A por fold e ponto de operação
└── plots/         ROC, PR, matriz de confusão e curvas de loss

results/<MODELO>/pd_table/     (ou results/comparison/pd_table/ ao comparar modelos)
├── pd_table_<ponto>.tex e .html       o tabelão de cada ponto de operação
├── pd_table_integrated.tex e .html    uma linha por modelo, juntando todas as regiões
└── pd_table_long.csv                  os números por trás das tabelas
```

---

## 📊 Como ler o tabelão

- As linhas são as regiões de $|\eta|$ e as colunas, as regiões de $E_T$. Cada célula traz
  $P_D$, $SP$ e $F_A$, com a média e o desvio entre os folds.
- Cada rede é ajustada para acertar exatamente o $P_D$ alvo (a coluna em verde). O que
  diferencia os modelos é o $SP$ e o $F_A$.
- A tabela integrada junta todas as regiões, e cada região pesa de acordo com o seu número de
  eventos.
- Para comparar modelos de forma justa, os configs precisam usar os mesmos dados,
  `max_files`, `n_splits` e `seed`. O `report` avisa quando isso não acontece.
- O `.tex` precisa de `\usepackage{booktabs}`, `\usepackage[table]{xcolor}` e
  `\usepackage{graphicx}`.

---

## 💡 Bom saber

- **Classes desbalanceadas:** em vez de descartar dados, a loss dá mais peso ao sinal. O peso
  é o número de eventos de ruído dividido pelo de sinal, calculado só nos dados de treino de
  cada fold.
- **Quando o treino para:** o early stopping acompanha o SP, usando o melhor corte possível em
  cada época.
- **A avaliação usa a região inteira**, incluindo os eventos com que cada fold treinou. Para
  olhar só os eventos que o fold não viu, use a coluna `in_sample` dos arquivos em `scores/`:

  ```python
  d = pd.read_parquet("results/MLP/et2_eta0/scores/fold_1.parquet")
  fora = d[~d.in_sample]
  ```

- **`n_splits: 1`** treina um modelo só, guardando 20% dos dados para a validação.

---

## 🧹 Limpeza

```bash
make clean
```
