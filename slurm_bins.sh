#!/bin/bash

# ==============================================================================
# Orquestrador Paralelo da Grade Et x Eta (5x5 = 25 redes) via SLURM
# ==============================================================================
# Cada bin vira 1 job na partition de GPU, que TREINA todos os folds daquele bin
# e em seguida AVALIA (métricas, gráficos e a fatia do tabelão daquela região).
# Quando os 25 bins terminam, um job final monta o tabelão completo.
# Ajuste PARTITION/RESERVATION abaixo se mudar.

PARTITION="gpu"
RESERVATION="gdi"

# Parâmetros de entrada com valores padrão
CONFIG_FILE=${1:-"ai/configs/mlp.yaml"}

N_ET_BINS=5
N_ETA_BINS=5

echo "====================================================================="
echo "Iniciando submissão da grade Et x Eta via SLURM (partition: $PARTITION)"
echo "Grade: ${N_ET_BINS}x${N_ETA_BINS} = $((N_ET_BINS * N_ETA_BINS)) redes"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "====================================================================="

# Cria pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="results/logs_slurm"
mkdir -p "$LOG_DIR"

# Loop disparando 1 sbatch para CADA bin (et, eta); o SLURM decide o nó/GPU.
JOB_IDS=()
for (( et=0; et<N_ET_BINS; et++ )); do
    for (( eta=0; eta<N_ETA_BINS; eta++ )); do

        echo "Submetendo bin et${et}_eta${eta}..."

        JOB_ID=$(sbatch --parsable -p "$PARTITION" --reservation="$RESERVATION" -N 1 --gres=gpu:1 \
             --job-name="et${et}_eta${eta}" \
             --output="${LOG_DIR}/et${et}_eta${eta}_%j.out" \
             --error="${LOG_DIR}/et${et}_eta${eta}_%j.err" \
             --wrap="python ai/run.py train --config $CONFIG_FILE --et-bin $et --eta-bin $eta && \
                     python ai/run.py evaluate --config $CONFIG_FILE --et-bin $et --eta-bin $eta")

        JOB_IDS+=("$JOB_ID")
        # Pequeno delay apenas para evitar concorrência no gerenciador de filas
        sleep 1
    done
done

# O tabelão agrega as 25 regiões, então só pode rodar depois que todas terminarem.
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
echo "Submetendo montagem do tabelão (após os ${#JOB_IDS[@]} bins)..."

REPORT_JOB_ID=$(sbatch --parsable -p "$PARTITION" --reservation="$RESERVATION" -N 1 \
     --dependency="afterok:${DEPENDENCY}" \
     --job-name="report" \
     --output="${LOG_DIR}/report_%j.out" \
     --error="${LOG_DIR}/report_%j.err" \
     --wrap="python ai/run.py report --config $CONFIG_FILE")

echo "====================================================================="
echo "Todas as ${#JOB_IDS[@]} redes da grade Et x Eta foram submetidas para a fila!"
echo "Tabelão enfileirado como job ${REPORT_JOB_ID}."
echo "Use 'squeue -u \$USER' para monitorar o andamento."
echo "Para cancelar tudo, rode 'scancel -u \$USER'."
echo "====================================================================="
echo "Resultados por região em 'results/<MODEL>/et<N>_eta<M>/', tabelão em"
echo "'results/<MODEL>/tabelao/' e logs em '${LOG_DIR}/'."
