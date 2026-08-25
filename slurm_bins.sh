#!/bin/bash

# ==============================================================================
# Orquestrador Paralelo da Grade Et x Eta (5x5 = 25 redes) via SLURM
# ==============================================================================
# Cada bin vira 1 job na partition de GPU; o próprio SLURM escalona entre os nós
# disponíveis (sem fixar hostname). Ajuste PARTITION/RESERVATION abaixo se mudar.

PARTITION="gpu"
RESERVATION="gdi"

# Parâmetros de entrada com valores padrão
CONFIG_FILE=${1:-"config.yaml"}

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
job=0
for (( et=0; et<N_ET_BINS; et++ )); do
    for (( eta=0; eta<N_ETA_BINS; eta++ )); do

        echo "Submetendo bin et${et}_eta${eta}..."

        sbatch -p "$PARTITION" --reservation="$RESERVATION" -N 1 --gres=gpu:1 \
             --job-name="MLP_Cern_et${et}_eta${eta}" \
             --output="${LOG_DIR}/et${et}_eta${eta}_%j.out" \
             --error="${LOG_DIR}/et${et}_eta${eta}_%j.err" \
             --wrap="python ai/run.py --config $CONFIG_FILE --et-bin $et --eta-bin $eta"

        job=$(( job + 1 ))
        # Pequeno delay apenas para evitar concorrência no gerenciador de filas
        sleep 1
    done
done

echo "====================================================================="
echo "Todas as $job redes da grade Et x Eta foram submetidas para a fila!"
echo "Use 'squeue -u \$USER' para monitorar o andamento."
echo "Para cancelar tudo, rode 'scancel -u \$USER'."
echo "====================================================================="
echo "Confira os resultados em 'results/<MODEL>/et<N>_eta<M>/' e os logs em '${LOG_DIR}/' quando acabarem."
