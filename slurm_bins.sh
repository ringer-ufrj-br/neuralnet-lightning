#!/bin/bash

# ==============================================================================
# Orquestrador da Grade Et x Eta (5x5 = 25 redes) via SLURM job array
# ==============================================================================
# A grade inteira vira UM job array de 25 tarefas: cada tarefa deriva seu par
# (et, eta) do SLURM_ARRAY_TASK_ID, TREINA todos os folds daquele bin e em
# seguida AVALIA (métricas, gráficos e a fatia do tabelão daquela região).
# Quando o array termina, um job final monta o tabelão completo.

# A partition 'gpu' e a 'cpu' sao filas distintas; sem -p o sbatch cai na default (cpu),
# que nao tem placa. Neste cluster as GPUs NAO estao registradas como GRES no SLURM
# (scontrol show node calobaXX -> Gres=(null)), entao pedir --gres=gpu:1 faz o sbatch
# recusar a submissao com "Requested node configuration is not available": escolher a
# partition gpu ja e o que garante um no com placa. Se um dia o admin configurar o GRES,
# basta rodar com GRES=gpu:1.
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-}"

# Parâmetros de entrada com valores padrão
CONFIG_FILE=${1:-"ai/configs/mlp.yaml"}
MAX_CONCURRENT=${2:-}          # opcional: limita quantas tarefas rodam ao mesmo tempo

N_ET_BINS=5
N_ETA_BINS=5
N_TASKS=$(( N_ET_BINS * N_ETA_BINS ))

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

# Índices do array: 0..24, opcionalmente com throttle (%N) para não tomar todas as GPUs.
ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
if [ -n "$MAX_CONCURRENT" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

# Array vazio quando GRES nao esta setado: assim o --gres nem chega na linha de comando.
GRES_FLAG=()
if [ -n "$GRES" ]; then
    GRES_FLAG=(--gres="$GRES")
fi

echo "====================================================================="
echo "Submetendo a grade Et x Eta como job array"
echo "Grade: ${N_ET_BINS}x${N_ETA_BINS} = ${N_TASKS} redes (array ${ARRAY_SPEC})"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "Partition: $PARTITION${GRES:+ (gres: $GRES)}"
echo "Interpretador: $PYTHON"
echo "====================================================================="

# Cria pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="$REPO_DIR/results/logs_slurm"
mkdir -p "$LOG_DIR"

# O \$ é escapado de propósito: SLURM_ARRAY_TASK_ID só existe quando a tarefa roda no nó,
# então a aritmética que mapeia o índice para (et, eta) precisa ser avaliada lá, não aqui.
ARRAY_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 "${GRES_FLAG[@]}" \
     --array="$ARRAY_SPEC" \
     --job-name="et_eta_grid" \
     --output="${LOG_DIR}/grid_%A_%a.out" \
     --error="${LOG_DIR}/grid_%A_%a.err" \
     --wrap="et=\$(( SLURM_ARRAY_TASK_ID / $N_ETA_BINS )); \
             eta=\$(( SLURM_ARRAY_TASK_ID % $N_ETA_BINS )); \
             echo \"Task \$SLURM_ARRAY_TASK_ID -> et\${et}_eta\${eta}\"; \
             $PYTHON ai/run.py train --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta && \
             $PYTHON ai/run.py evaluate --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta") || {
    echo "ERRO: sbatch recusou o job array; nada foi enfileirado." >&2
    exit 1
}

# O tabelão agrega as 25 regiões, então só pode rodar depois que o array inteiro terminar.
# 'afterok:<id do array>' espera todas as tarefas concluírem com sucesso.
REPORT_JOB_ID=$(sbatch --parsable --chdir="$REPO_DIR" -N 1 \
     --dependency="afterok:${ARRAY_JOB_ID}" \
     --job-name="report" \
     --output="${LOG_DIR}/report_%j.out" \
     --error="${LOG_DIR}/report_%j.err" \
     --wrap="$PYTHON ai/run.py report --config $CONFIG_FILE") || {
    echo "ERRO: sbatch recusou o job do tabelão; cancele a grade com 'scancel ${ARRAY_JOB_ID}'" >&2
    echo "      ou rode o tabelão na mão depois: $PYTHON ai/run.py report --config $CONFIG_FILE" >&2
    exit 1
}

echo "====================================================================="
echo "Job array ${ARRAY_JOB_ID} submetido (${N_TASKS} tarefas)."
echo "Tabelão enfileirado como job ${REPORT_JOB_ID}."
echo "Use 'squeue -u \$USER' para monitorar e 'scancel ${ARRAY_JOB_ID}' para cancelar a grade."
echo "Uma tarefa isolada: 'scancel ${ARRAY_JOB_ID}_<indice>'."
echo "====================================================================="
echo "Resultados por região em 'results/<MODEL>/et<N>_eta<M>/', tabelão em"
echo "'results/<MODEL>/pd_table/' e logs em '${LOG_DIR}/'."
