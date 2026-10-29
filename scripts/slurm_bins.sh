#!/bin/bash
# ==============================================================================
# Grade Et x Eta via SLURM: UM TREINO POR TAREFA
# ==============================================================================
#   ./scripts/slurm_bins.sh <config> [max_concorrentes]
#
# Três etapas encadeadas por dependência (afterok), cada uma começa quando a anterior termina
# inteira:
#
#   1. array de n_regioes x n_folds x n_inits tarefas -> um treino cada
#   2. array de n_regioes tarefas -> avalia a região (usa o melhor init de cada fold; todos
#      os inits continuam salvos)
#   3. um job -> monta o tabelão
#
# Variáveis de ambiente: PARTITION (default gpu), GRES (vazio: as GPUs deste cluster não estão
# registradas como GRES; a partition gpu já garante um nó com placa), PYTHON (default
# neuralnet-env/bin/python do repositório). Para as MLPs em CPU, veja scripts/README.md.
set -euo pipefail

PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-}"

CONFIG_FILE=${1:?"uso: $0 <config> [max_concorrentes]"}
MAX_CONCURRENT=${2:-}

# O job roda no nó de computação, que não herda de forma confiável o ambiente do nó de login:
# o interpretador é resolvido por caminho absoluto e o diretório de trabalho é fixado no
# repositório (caminhos relativos do config são relativos a ele).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/neuralnet-env/bin/python}"
cd "$REPO_DIR"

if [ ! -x "$PYTHON" ]; then
    echo "ERRO: interpretador não encontrado em '$PYTHON'." >&2
    echo "      Crie o ambiente com 'make venv' (dentro de uma alocação, não no nó de login)" >&2
    echo "      ou indique outro com PYTHON=/caminho/para/python $0 ..." >&2
    exit 1
fi

# A grade é fixa (neuralnet_lightning/binning.py); n_splits, n_inits e o destino saem do config, que é
# validado aqui - um erro de digitação falha agora, e não em 625 tarefas.
read -r N_ET_BINS N_ETA_BINS N_FOLDS N_INITS RESULTS NAME < <("$PYTHON" -c "
import sys
from neuralnet_lightning.binning import N_ET_BINS, N_ETA_BINS
from neuralnet_lightning.config import load_config
c = load_config(sys.argv[1])
print(N_ET_BINS, N_ETA_BINS, c.training.n_splits, c.training.n_inits, c.results_root, c.name)
" "$CONFIG_FILE") || { echo "ERRO: config inválido: '$CONFIG_FILE'." >&2; exit 1; }

N_BINS=$(( N_ET_BINS * N_ETA_BINS ))
N_TASKS=$(( N_BINS * N_FOLDS * N_INITS ))

ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
if [ -n "$MAX_CONCURRENT" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

GRES_FLAG=()
if [ -n "$GRES" ]; then
    GRES_FLAG=(--gres="$GRES")
fi

# results_root pode ser relativo ao repositório ou absoluto.
case "$RESULTS" in
    /*) EXPERIMENT_DIR="$RESULTS/$NAME" ;;
    *)  EXPERIMENT_DIR="$REPO_DIR/$RESULTS/$NAME" ;;
esac
LOG_DIR="$EXPERIMENT_DIR/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$PYTHON -m neuralnet_lightning"

echo "====================================================================="
echo "Experimento: $NAME ($CONFIG_FILE)"
echo "Grade: ${N_ET_BINS}x${N_ETA_BINS} = ${N_BINS} regiões"
echo "Treinos: ${N_BINS} x ${N_FOLDS} folds x ${N_INITS} inits = ${N_TASKS} (array ${ARRAY_SPEC})"
echo "Partition: $PARTITION${GRES:+ (gres: $GRES)} | Interpretador: $PYTHON"
echo "====================================================================="

# Etapa 1. O \$ é escapado de propósito: SLURM_ARRAY_TASK_ID só existe no nó, então a
# decomposição do índice é avaliada lá. É a inversa de
#   id = ((et * N_ETA + eta) * N_FOLDS + (fold-1)) * N_INITS + (init-1)
# de modo que tarefas da mesma região ficam contíguas e um throttle (%N) mantém juntas as que
# compartilham o mesmo carregamento de dados.
TRAIN_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 "${GRES_FLAG[@]}" \
     --array="$ARRAY_SPEC" \
     --job-name="train_${NAME}" \
     --output="${LOG_DIR}/train_%A_%a.out" \
     --error="${LOG_DIR}/train_%A_%a.err" \
     --wrap="i=\$SLURM_ARRAY_TASK_ID; \
             init=\$(( i % $N_INITS + 1 )); i=\$(( i / $N_INITS )); \
             fold=\$(( i % $N_FOLDS + 1 )); i=\$(( i / $N_FOLDS )); \
             eta=\$(( i % $N_ETA_BINS )); et=\$(( i / $N_ETA_BINS )); \
             echo \"Task \$SLURM_ARRAY_TASK_ID -> et\${et}_eta\${eta} fold \${fold} init \${init}\"; \
             $RUN train --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta --fold \$fold --init \$init") || {
    echo "ERRO: sbatch recusou o array de treino; nada foi enfileirado." >&2
    exit 1
}

# Etapa 2: só pode rodar quando TODOS os inits terminaram, porque o evaluate compara os
# n_inits checkpoints de cada fold.
EVAL_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 "${GRES_FLAG[@]}" \
     --array="0-$(( N_BINS - 1 ))" \
     --dependency="afterok:${TRAIN_JOB_ID}" \
     --job-name="eval_${NAME}" \
     --output="${LOG_DIR}/eval_%A_%a.out" \
     --error="${LOG_DIR}/eval_%A_%a.err" \
     --wrap="et=\$(( SLURM_ARRAY_TASK_ID / $N_ETA_BINS )); eta=\$(( SLURM_ARRAY_TASK_ID % $N_ETA_BINS )); \
             echo \"Task \$SLURM_ARRAY_TASK_ID -> et\${et}_eta\${eta}\"; \
             $RUN evaluate --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta") || {
    echo "ERRO: sbatch recusou o array de avaliação; cancele o treino com 'scancel ${TRAIN_JOB_ID}'." >&2
    exit 1
}

# Etapa 3: o tabelão agrega todas as regiões.
REPORT_JOB_ID=$(sbatch --parsable --chdir="$REPO_DIR" -N 1 \
     --dependency="afterok:${EVAL_JOB_ID}" \
     --job-name="report_${NAME}" \
     --output="${LOG_DIR}/report_%j.out" \
     --error="${LOG_DIR}/report_%j.err" \
     --wrap="$RUN report --config $CONFIG_FILE") || {
    echo "ERRO: sbatch recusou o job do tabelão; cancele com 'scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID}'" >&2
    echo "      ou rode depois: $RUN report --config $CONFIG_FILE" >&2
    exit 1
}

echo "Treino:    array ${TRAIN_JOB_ID} (${N_TASKS} tarefas, um treino cada)"
echo "Avaliação: array ${EVAL_JOB_ID} (${N_BINS} tarefas), após o treino"
echo "Tabelão:   job ${REPORT_JOB_ID}, após a avaliação"
echo "Cancelar tudo: scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID} ${REPORT_JOB_ID}"
echo "Resultados em '$EXPERIMENT_DIR/et<N>_eta<M>/', tabelão em '$EXPERIMENT_DIR/pd_table/', logs em '$LOG_DIR/'."
