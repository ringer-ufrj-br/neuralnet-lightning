#!/bin/bash
# Roda a grade Et x Eta inteira nesta máquina, sem SLURM: treina e avalia uma região de cada
# vez e monta o tabelão no final.
#
#   ./scripts/run_local_grid.sh [config]
#
# Uma região já avaliada (com metrics/folds_long.csv, o mesmo critério do `report --list`) é
# pulada, então uma rodada interrompida continua de onde parou. FORCE=1 refaz tudo.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/neuralnet-env/bin/python}"
CONFIG=${1:-ai/configs/mlp.yaml}
cd "$REPO_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "ERRO: interpretador não encontrado em '$PYTHON'." >&2
    echo "      Crie o ambiente com 'make venv' ou indique outro com PYTHON=/caminho/para/python $0 ..." >&2
    exit 1
fi

# Onde o evaluate grava cada região: <results_root>/<model>/et<N>_eta<M>, com os mesmos
# defaults do ai/run.py.
read -r RESULTS MODEL < <("$PYTHON" -c "
import sys
from ai.run import load_config
c = load_config(sys.argv[1])
print(c.get('results_root', 'results'), c.get('model', 'CNN2D'))
" "$CONFIG")

# As regiões chegam pelo fd 3, para que o treino e o evaluate não leiam a lista pelo stdin.
while read -r et eta <&3; do
    region="et${et}_eta${eta}"
    if [ "${FORCE:-0}" != 1 ] && [ -f "$RESULTS/$MODEL/$region/metrics/folds_long.csv" ]; then
        echo "==> $region já avaliada, pulando."
        continue
    fi
    echo "==> $region"
    "$PYTHON" ai/run.py train    --config "$CONFIG" --et-bin "$et" --eta-bin "$eta"
    "$PYTHON" ai/run.py evaluate --config "$CONFIG" --et-bin "$et" --eta-bin "$eta"
done 3< <("$PYTHON" ai/run.py grid)

"$PYTHON" ai/run.py report --config "$CONFIG"
