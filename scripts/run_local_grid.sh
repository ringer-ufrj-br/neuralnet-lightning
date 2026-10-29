#!/bin/bash
# Roda a grade Et x Eta inteira nesta máquina, sem SLURM: treina e avalia uma região de cada
# vez e monta o tabelão no final.
#
#   ./scripts/run_local_grid.sh <config>
#
# Uma região já avaliada (com metrics/folds_long.csv, o mesmo critério do `report --list`) é
# pulada, então uma rodada interrompida continua de onde parou. FORCE=1 refaz tudo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/neuralnet-env/bin/python}"
CONFIG=${1:?"uso: $0 <config>"}
cd "$REPO_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "ERRO: interpretador não encontrado em '$PYTHON'." >&2
    echo "      Crie o ambiente com 'make venv' ou indique outro com PYTHON=/caminho/para/python $0 ..." >&2
    exit 1
fi

# Onde o evaluate grava cada região: <results_root>/<name>/et<N>_eta<M> (neuralnet_lightning/paths.py).
read -r RESULTS NAME < <("$PYTHON" -c "
import sys
from neuralnet_lightning.config import load_config
c = load_config(sys.argv[1])
print(c.results_root, c.name)
" "$CONFIG")

# As regiões chegam pelo fd 3, para que o treino e o evaluate não leiam a lista pelo stdin.
while read -r et eta <&3; do
    region="et${et}_eta${eta}"
    if [ "${FORCE:-0}" != 1 ] && [ -f "$RESULTS/$NAME/$region/metrics/folds_long.csv" ]; then
        echo "==> $region já avaliada, pulando."
        continue
    fi
    echo "==> $region"
    "$PYTHON" -m neuralnet_lightning train    --config "$CONFIG" --et-bin "$et" --eta-bin "$eta"
    "$PYTHON" -m neuralnet_lightning evaluate --config "$CONFIG" --et-bin "$et" --eta-bin "$eta"
done 3< <("$PYTHON" -m neuralnet_lightning grid)

"$PYTHON" -m neuralnet_lightning report --config "$CONFIG"
