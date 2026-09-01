#!/bin/bash

# ==============================================================================
# Orquestrador Paralelo de Validação Cruzada (K-Fold) via SLURM
# ==============================================================================
# Cada fold vira 1 job de TREINO na partition de GPU; o próprio SLURM escalona
# entre os nós disponíveis (sem fixar hostname). Quando todos os folds terminam,
# um job de AVALIAÇÃO roda automaticamente (--dependency=afterok) e produz as
# métricas, os gráficos e a fatia do tabelão desta região.
# Ajuste PARTITION abaixo se mudar.

PARTITION="gpu"

# Parâmetros de entrada com valores padrão
NUM_FOLDS=${1:-5}           # Padrão: rodar 5 folds (deve bater com n_splits do yaml)
CONFIG_FILE=${2:-"ai/configs/mlp.yaml"}

# O job roda no nó de computação, que não herda confiavelmente o ambiente do nó de login:
# resolvemos o interpretador do venv por caminho absoluto e fixamos o diretório de trabalho
# do job no repositório (os caminhos de config, data/ e results/ são todos relativos a ele).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/neuralnet-env/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERRO: interpretador nao encontrado em '$PYTHON'." >&2
    echo "      Crie o ambiente com 'make venv' (de dentro de uma alocacao, nao no no de login)" >&2
    echo "      ou aponte outro com PYTHON=/caminho/para/python $0 ..." >&2
    exit 1
fi

echo "====================================================================="
echo "Iniciando submissão paralela via SLURM (partition: $PARTITION)"
echo "Folds a calcular: $NUM_FOLDS"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "====================================================================="

# Cria pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="results/logs_slurm"
mkdir -p "$LOG_DIR"

# Loop disparando 1 sbatch para CADA fold; o SLURM decide o nó/GPU.
TRAIN_JOB_IDS=()
for (( fold=1; fold<=NUM_FOLDS; fold++ )); do

    echo "Submetendo treino do Fold $fold..."

    JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 --gres=gpu:1 \
         --job-name="train_fold_${fold}" \
         --output="${LOG_DIR}/train_fold_${fold}_%j.out" \
         --error="${LOG_DIR}/train_fold_${fold}_%j.err" \
         --wrap="$PYTHON ai/run.py train --config $CONFIG_FILE --fold $fold")

    TRAIN_JOB_IDS+=("$JOB_ID")

    # Pequeno delay apenas para evitar concorrência no gerenciador de filas
    sleep 1
done

# Avaliação depende de TODOS os folds: só faz sentido comparar folds quando todos
# os checkpoints existem (o tabelão é média ± desvio entre eles).
DEPENDENCY=$(IFS=:; echo "${TRAIN_JOB_IDS[*]}")
echo "Submetendo avaliação (após os folds ${DEPENDENCY})..."

EVAL_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 --gres=gpu:1 \
     --dependency="afterok:${DEPENDENCY}" \
     --job-name="evaluate" \
     --output="${LOG_DIR}/evaluate_%j.out" \
     --error="${LOG_DIR}/evaluate_%j.err" \
     --wrap="$PYTHON ai/run.py evaluate --config $CONFIG_FILE && $PYTHON ai/run.py report --config $CONFIG_FILE")

echo "====================================================================="
echo "Todos os $NUM_FOLDS folds foram submetidos para a fila!"
echo "Avaliação + tabelão enfileirados como job ${EVAL_JOB_ID}."
echo "Use 'squeue -u \$USER' para monitorar o andamento."
echo "Para cancelar tudo, rode 'scancel -u \$USER'."
echo "====================================================================="
echo "Confira os resultados em 'results/' e os logs em '${LOG_DIR}/' quando acabarem."
