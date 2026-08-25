#!/bin/bash

# ==============================================================================
# Orquestrador Paralelo de Validação Cruzada (K-Fold) via SLURM
# ==============================================================================
# Cada fold vira 1 job na partition de GPU; o próprio SLURM escalona entre os nós
# disponíveis (sem fixar hostname). Ajuste PARTITION/RESERVATION abaixo se mudar.

PARTITION="gpu"
RESERVATION="gdi"

# Parâmetros de entrada com valores padrão
NUM_FOLDS=${1:-3}           # Padrão: rodar 3 folds (deve bater com o yaml)
CONFIG_FILE=${2:-"config.yaml"}

echo "====================================================================="
echo "Iniciando submissão paralela via SLURM (partition: $PARTITION)"
echo "Folds a calcular: $NUM_FOLDS"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "====================================================================="

# Cria pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="results/logs_slurm"
mkdir -p "$LOG_DIR"

# Loop disparando 1 sbatch para CADA fold; o SLURM decide o nó/GPU.
for (( fold=1; fold<=NUM_FOLDS; fold++ )); do

    echo "Submetendo Fold $fold..."

    sbatch -p "$PARTITION" --reservation="$RESERVATION" -N 1 --gres=gpu:1 \
         --job-name="CNN_2D_Cern_fold_${fold}" \
         --output="${LOG_DIR}/fold_${fold}_%j.out" \
         --error="${LOG_DIR}/fold_${fold}_%j.err" \
         --wrap="python ai/run.py --config $CONFIG_FILE --fold $fold"

    # Pequeno delay apenas para evitar concorrência no gerenciador de filas
    sleep 1
done

echo "====================================================================="
echo "Todos os $NUM_FOLDS folds foram submetidos para a fila!"
echo "Use 'squeue -u \$USER' para monitorar o andamento."
echo "Para cancelar tudo, rode 'scancel -u \$USER'."
echo "====================================================================="
echo "Confira os gráficos em 'results/' e os logs em '${LOG_DIR}/' quando acabarem."
