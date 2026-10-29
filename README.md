# neuralnet-lightning

Neural networks (PyTorch Lightning) for the ATLAS Ringer: electron × jet classification with
cross-validation, one network per $E_T$ × $|\eta|$ region, metrics, plots and the final
efficiency table (the "tabelão").

Each experiment is a YAML file that combines a model and a dataset. A new dataset is a YAML
file; a new model is a Python file.

---

## 📌 How it works

| Command | What it does |
|---|---|
| `train` | Trains the network with cross-validation and saves every model (each fold × initialisation) |
| `evaluate` | Picks each fold's best initialisation, scores the region and writes metrics and plots |
| `report` | Gathers every evaluated region into the efficiency table (LaTeX and HTML) |
| `grid` | Prints the regions of the grid, for scripts that distribute the work |

Each step reads what the previous one saved to disk, so evaluation and the table can be redone
without retraining.

---

## ⚙️ Installation

```bash
make venv                            # creates neuralnet-env and installs the package with its test extras
source neuralnet-env/bin/activate
make test                            # ~1-2 min on CPU, on synthetic data
```

Versions are pinned in `pyproject.toml`. `requirements.txt` only adds the package index for
`torch==2.13.0+cu126`: the GPU nodes run driver 550 (CUDA 12.4), and the default PyPI `torch`
needs driver 580+.

**On the cluster**, run `make venv` inside an allocation, not on the login node (installing
torch is heavy). For example:

```bash
srun -p gpu --pty bash
make venv && ./neuralnet-env/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Keep the environment inside the repository folder so the compute nodes can see it.

**The data** is read directly from the shared storage (paths in `configs/datasets/`). For a
local copy, use `make copy-data` (`DATA_SRC=/other/path` to copy another folder) and point
`data_path` to it.

---

## ⚡ Quick start

Train, evaluate and build the table for a single region:

```bash
python -m neuralnet_lightning train    --config configs/mlp_mc21.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning evaluate --config configs/mlp_mc21.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning report   --config configs/mlp_mc21.yaml
```

Results go to `results/mlp_mc21/et2_eta0/`, the table to `results/mlp_mc21/pd_table/`. To run
the whole grid, see [Running](#-running).

---

## 🧠 Models

| `model:` | Description | Datasets |
|---|---|---|
| `MLP` | One hidden layer (`hidden`, default 5) over the inner half of each layer's rings (50), norm1-normalised (each event divided by the sum of its rings) | mc21, mc25 |
| `CNN2D` | Convolutional network over the calorimeter cell images (one layer = one channel) | mc25 |
| `Fused` | Rings and cell images in two branches, joined by a fusion head | mc25 |

`CNN2D` and `Fused` need the cell images (`cl_cells_*`), which only mc25 has. On a dataset
without them, `train` stops while loading with
`❌ Column(s) ['cl_cells_presampler', ...] are not in the dataset`.

---

## 📝 Configuration

An experiment:

```yaml
name: mlp_mc21                # results go to <results_root>/<name>/<region>/
model: MLP                    # the architecture (table above)
results_root: results         # optional

dataset: datasets/mc21.yaml   # a file in configs/datasets/ (relative to this config)

model_args:                   # optional: overrides the model's build_network defaults
  hidden: 5

training:                     # all optional; the values below are the defaults
  max_epochs: 5000
  batch_size: 1024
  learning_rate: 0.001
  patience: 50                # epochs without SP improvement before stopping
  n_splits: 10                # folds (1 = a single model with a 20% validation split)
  n_inits: 5                  # initialisations per fold; evaluate uses the best one
  seed: 42                    # seed of the fold split
  accelerator: auto           # auto, cpu, gpu
  devices: auto

evaluation:
  operating_points: {tight: 0.90, medium: 0.95, loose: 0.99}   # name -> target P_D
```

A dataset (`configs/datasets/mc21.yaml`):

```yaml
data_path: /mnt/shared/.../mc21_isabela_qt_2sigma_restriction/electron_ringer.parquet
rings_col: "TrigEMClusterContainer.ringsE"   # a list column holding the 100 rings,
                                             # or "cl_ring_%i": one column per ring
et_col: "TrigEMClusterContainer.et"          # in MeV
eta_col: "TrigEMClusterContainer.eta"
label_col: target             # if omitted, the label comes from the file name (Zee = 1, JF17 = 0)
max_files: 10                 # optional: only the first N files of each folder
```

An experiment's `dataset:` accepts three forms:

```yaml
dataset: datasets/mc25.yaml                                           # the file
dataset: {file: datasets/mc25.yaml, max_files: 100}                   # the file, overriding some keys
dataset: {data_path: ..., rings_col: ..., et_col: ..., eta_col: ...}  # everything inline
```

- The `file` path is relative to the folder of the config that references it, so commands can
  run from any directory.
- `data_path`, `rings_col`, `et_col` and `eta_col` are required; there are no default column
  names in the code.
- The config is validated on load: an unknown key (such as `bacth_size`) or a value of the
  wrong type is an error, not silently ignored. The defaults live only in the dataclasses of
  `neuralnet_lightning/config.py`.
- `name` and `model` are independent. The same model on another dataset, or with other
  `model_args`, is just another config with another `name`.

Shipped configs:

| Config | `name` | Model | Dataset | Folds × inits |
|---|---|---|---|---|
| `configs/mlp.yaml` | `mlp_mc25` | `MLP` | mc25 | 10 × 5 |
| `configs/mlp_mc21.yaml` | `mlp_mc21` | `MLP` | mc21 | 5 × 5 |
| `configs/cnn2d.yaml` | `cnn2d_mc25` | `CNN2D` | mc25, 100 files per folder | 5 × 1 |
| `configs/fused.yaml` | `fused_mc25` | `Fused` | mc25, 100 files per folder | 5 × 1 |

---

## 🚀 Running

For one region:

```bash
python -m neuralnet_lightning train    --config configs/mlp.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning evaluate --config configs/mlp.yaml --et-bin 2 --eta-bin 0
python -m neuralnet_lightning report   --config configs/mlp.yaml
```

Without `--et-bin`/`--eta-bin`, a single network covers the whole phase space (`<name>/all/`).

Useful options:

| Option | Effect |
|---|---|
| `train --fold N --init M` | Trains a single model, as each SLURM task does |
| `train --accelerator cpu --devices 1` | Overrides the config |
| `evaluate --reuse-scores` | Recomputes metrics and plots without running the network again |
| `evaluate --no-plots` | Computes the metrics only |
| `report --names cnn2d_mc25,fused_mc25` | Compares experiments, rows in this order |
| `report --list` | Shows what has been trained and evaluated, and the command that fills each gap |
| `report --no-integrated` | Skips the integrated table |

### The whole grid on your machine

```bash
./scripts/run_local_grid.sh configs/mlp.yaml
```

Trains and evaluates the 25 regions one at a time, then builds the table. Regions already
evaluated are skipped (`FORCE=1` redoes everything).

### The whole grid on the cluster (SLURM)

```bash
./scripts/slurm_bins.sh configs/mlp.yaml       # the whole grid
./scripts/slurm_bins.sh configs/mlp.yaml 40    # at most 40 tasks at a time
```

Run it from the login node. The script validates the config first (a typo fails immediately,
not in 625 tasks) and submits three stages, each waiting for the previous one to succeed:

1. a job array with one training per task (region × fold × initialisation);
2. a job array with one evaluation per region, using each fold's best initialisation;
3. one job that builds the table.

Environment variables: `PARTITION` (default `gpu`), `GRES`, `PYTHON` (default
`neuralnet-env/bin/python`). Logs go to `<results_root>/<name>/slurm_logs/`; cancel with
`scancel <id>`.

> **MLPs run best on CPU.** Read [scripts/README.md](scripts/README.md) before submitting them:
> without `SBATCH_MEM_PER_NODE` and `OMP_NUM_THREADS=1`, each task takes a whole node.

---

## 🗃️ Adding a dataset

A new dataset is a file in `configs/datasets/`. For ⟨μ⟩ = 200, for example:

1. Inspect the columns of one file:

   ```python
   import polars as pl
   print(pl.read_parquet_schema("/path/to/mu200/00000000.parquet"))
   ```

2. Create `configs/datasets/mu200.yaml` with what you found:

   ```yaml
   data_path: /path/to/mu200               # folder, file or glob; subfolders are read
   rings_col: "TrigEMClusterContainer.ringsE"
   et_col: "TrigEMClusterContainer.et"     # in MeV (the E_T grid is in MeV)
   eta_col: "TrigEMClusterContainer.eta"
   label_col: target                       # or omit it if the label comes from the file name
   ```

3. Point an experiment at it, under a new `name`:

   ```yaml
   # configs/mlp_mu200.yaml
   name: mlp_mu200
   model: MLP
   dataset: datasets/mu200.yaml
   training: {...}                          # copy from mlp_mc21.yaml to compare
   ```

4. Try one region before the whole grid:

   ```bash
   python -m neuralnet_lightning train --config configs/mlp_mu200.yaml --et-bin 2 --eta-bin 0
   ```

`make test` checks that every file in `configs/` and `configs/datasets/` is valid.

If the label comes from the file name with patterns other than Zee/JF17, edit `PATTERNS` in
`neuralnet_lightning/data/labels.py`. A file matching no pattern is rejected rather than given
a wrong label.

---

## 🧩 Adding a model

A new model is **one file** in `neuralnet_lightning/models/`. It registers itself; there is no
list to edit:

```python
# neuralnet_lightning/models/my_net.py
import torch.nn as nn

from neuralnet_lightning.binary.model import BaseBinaryClassifier
from neuralnet_lightning.models.registry import register_model
from neuralnet_lightning.preprocess.mlp import PreprocessMLP


@register_model("MyNet")                     # the name used in `model:`
class MyNet(BaseBinaryClassifier):
    preprocessor = PreprocessMLP             # the input the network receives

    def build_network(self, input_dim: int = 50, hidden: int = 16, dropout: float = 0.1) -> nn.Module:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                             nn.Linear(hidden, 1))
```

- Don't write an `__init__`. The base class handles the loss (BCE with `pos_weight`), the
  metrics, the `val_sp` that early stopping monitors, and the optimizer.
- The arguments of `build_network` are the architecture's hyperparameters. They are saved in
  the checkpoint and can be overridden by the config's `model_args`. Values only known after
  preprocessing (such as `input_dim`) come from the preprocessor's `model_kwargs`.
- The network returns raw logits `(Batch, 1)`, without a sigmoid.
- If needed, override `forward` (multi-branch networks, see
  `neuralnet_lightning/models/fused.py`), `compute_loss` (auxiliary losses), `build_metrics`
  or `configure_optimizers`.

To list the registered models:

```bash
python -c "from neuralnet_lightning.models.registry import available_models; print(available_models())"
```

### A new input representation

If the model needs a different input, also write a preprocessor in
`neuralnet_lightning/preprocess/`. For example, rings + mc21 shower shapes:

```python
# neuralnet_lightning/preprocess/rings_ss.py
import numpy as np
import polars as pl

from neuralnet_lightning.preprocess.base import BasePreprocessor
from neuralnet_lightning.preprocess.mlp import selected_rings

SHOWER_SHAPES = ["HLTElectronContainer.reta", "HLTElectronContainer.eratio", "HLTElectronContainer.rhad"]


class PreprocessRingsSS(BasePreprocessor):
    def required_columns(self):
        return selected_rings() + SHOWER_SHAPES            # only these columns are read from disk

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        rings = self.normalize(self.extract(df, selected_rings()))   # norm1 on the rings only
        return np.hstack([rings, self.extract(df, SHOWER_SHAPES)])

    def model_kwargs(self, X):
        return {"input_dim": X.shape[1]}
```

- `df` is a polars DataFrame. Don't convert it to pandas: that copies the whole frame.
- If the preprocessor must learn something from the data (e.g. the mean and std of each shower
  shape), implement `fit`. The fitted object is saved at training time and reused by
  `evaluate`. Note that `fit` sees the whole region, before the fold split.
- A different normalisation is a different preprocessor (a subclass overriding `normalize`),
  so each checkpoint records which normalisation it was trained with.

**Shower shapes and tracking variables have no canonical names**: columns other than rings,
Et, eta, label and id keep their parquet names, which differ between datasets (`cl_rhad` in
mc25, `HLTElectronContainer.rhad` in mc21). The preprocessor above therefore only works on
datasets with those names. To share such a model across datasets, give the variables canonical
names in `DatasetSchema` (`neuralnet_lightning/data/schema.py`), as is done for the rings: a new
field (e.g. `shower_shape_cols`) automatically becomes a key of the `dataset:` block.

---

## ✂️ Modifying the dataset

Changes that apply to **every** model (a quality cut, a derived column, a new target) belong in
`neuralnet_lightning/data/loader.py`, as polars expressions on the `LazyFrame` before `collect`.
They then run inside the streaming scan, and only the rows and columns needed reach memory.
For example, a cut:

```python
# in load_dataframe, right after `frame = schema.scan(files)`:
frame = frame.filter(pl.col("TrigEMClusterContainer.isGoodRinger"))
```

Before the projection, the filter sees the original parquet column names, and the cut column is
not kept in the final DataFrame. To make the change configurable per dataset, add a field to
`DatasetSchema`; it becomes a key of the `dataset:` block.

Changes specific to **one** architecture (normalisation, feature selection) belong in its
preprocessor, in `neuralnet_lightning/preprocess/`.

---

## 📂 Output layout

```
results/<name>/<region>/          region = et<i>_eta<j>, or "all" without binning
├── artifacts/     preprocessor.joblib and the validation indices of each fold
├── checkpoints/   fold_N_init_M.ckpt and its .json sidecar with the training details
├── history/       per-epoch loss, per fold and initialisation
├── logs/          Lightning logs
├── scores/        fold_N.parquet: the network score of every event in the region (+ et, eta, in_sample)
├── metrics/       folds_long.csv: P_D, SP and F_A per fold and operating point
└── plots/         ROC, PR, confusion matrix, loss curves, per-fold ROC

results/<name>/pd_table/          (results/comparison/pd_table/ when comparing experiments)
├── pd_table_<point>.tex / .html       the table of each operating point
├── pd_table_integrated.tex / .html    one row per experiment, pooling every region
└── pd_table_long.csv                  the numbers behind the tables

results/<name>/slurm_logs/        SLURM task logs
```

---

## 📊 Reading the table

- Rows are $|\eta|$ regions and columns are $E_T$ regions. Each cell shows $P_D$, $SP$ and
  $F_A$ as mean ± std over the folds.
- Every network is tuned to the target $P_D$ (green column); what tells models apart is $SP$
  and $F_A$.
- The integrated table pools every region, each weighted by its number of events.
- A fair comparison between experiments needs the same data, `max_files`, `n_splits` and
  `seed`. `report` warns when they differ.
- The `.tex` needs `\usepackage{booktabs}`, `\usepackage[table]{xcolor}` and
  `\usepackage{graphicx}`.

---

## 🗺️ Code layout

```
neuralnet_lightning/
├── __main__.py      CLI
├── config.py        YAML dataclasses (the only defaults) and dataset files
├── binning.py       Et × |eta| grid and Region
├── paths.py         RunPaths: where each artefact of an (experiment, region) lives
├── pipeline.py      train / evaluate for one region: loader → preprocessor → training / evaluation
├── training.py      K-Fold, TensorBatchLoader, whole dataset on the GPU when it fits
├── evaluation.py    best initialisation per fold, inference, scores/*.parquet
├── data/            schema (canonical column names), labels from file names, loader
├── preprocess/      canonical DataFrame → model input (mlp, cnn2d, fused)
├── models/          registry + one file per architecture (mlp, cnn2d, fused)
└── binary/          electron × jet task: base model, SP index, operating points, plots, table
```

The core (data, config, registry, training, evaluation, CLI) does not assume a binary task. It
reads what it needs from the model class: the preprocessor, the monitored metric and its
direction, the per-fold kwargs (`pos_weight`), what to stratify on, and how to summarise the
scores. This contract is described in `neuralnet_lightning/models/registry.py`.

---

## 💡 Good to know

- **Class imbalance:** instead of discarding data, the loss uses `pos_weight` =
  background/signal, computed on each fold's training split only.
- **Early stopping** monitors the maximum validation $SP$ over every possible threshold, each
  epoch.
- **Evaluation covers the whole region**, including the events each fold trained on. To keep
  only the events a fold did not see, filter on `in_sample`:

  ```python
  import polars as pl
  scores = pl.read_parquet("results/mlp_mc21/et2_eta0/scores/fold_1.parquet")
  out_of_sample = scores.filter(~pl.col("in_sample"))
  ```

- **Reproducible:** row order is deterministic (sorted folders and files), the folds depend
  only on `seed`, and each (fold, init) has its own seed. Retraining a single (fold, init)
  gives the same model.
- **`evaluate` refuses to run** if the number of events in the region changed since training
  (different `max_files`, new files in the folder): the fold indices would no longer be valid.
- **Old results** produced before the restructuring (`results/<MODEL>/...`, with
  `fold_N.json`) are not read by the current code and must be regenerated.
- **Loading data in a notebook:**

  ```python
  from neuralnet_lightning.binning import Region
  from neuralnet_lightning.config import load_config
  from neuralnet_lightning.data.loader import load_dataframe

  cfg = load_config("configs/mlp_mc21.yaml")
  df = load_dataframe(cfg.dataset, ["ring_0", "ring_1"], Region(2, 0))
  ```

---

## 🧹 Cleanup

```bash
make clean    # removes the environment, caches and build metadata
```
