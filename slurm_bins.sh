#!/bin/bash

# ==============================================================================
# Orquestrador da Grade Et x Eta via SLURM (tamanho vem do config)
# ==============================================================================
# UM TREINO POR TAREFA. Cada tarefa do primeiro array deriva sua tripla
# (et, eta, fold, init) do SLURM_ARRAY_TASK_ID e treina exatamente um modelo,
# em vez de um bin inteiro (que eram n_splits x n_inits = 25 treinos numa
# tarefa so). Sao tres etapas encadeadas por dependencia:
#
#   1. array de n_bins x n_folds x n_inits tarefas -> um treino cada
#   2. array de n_bins tarefas -> escolhe o melhor init de cada fold (select)
#      e avalia a regiao
#   3. um job -> monta o tabelao
#
# Cada etapa so comeca quando a anterior termina inteira (afterok).

# As partitions 'gpu' e 'cpu' sao filas distintas; sem -p o sbatch usa a default (cpu),
# que nao tem placa. Neste cluster as GPUs NAO estao registradas como GRES no SLURM
# (scontrol show node calobaXX -> Gres=(null)), portanto --gres=gpu:1 faz o sbatch
# recusar a submissao com "Requested node configuration is not available": a escolha da
# partition gpu ja garante um no com placa. Caso o GRES venha a ser configurado,
# submeta com GRES=gpu:1.
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-}"

# Parâmetros de entrada com valores padrão
CONFIG_FILE=${1:-"ai/configs/mlp.yaml"}
MAX_CONCURRENT=${2:-}          # opcional: limita quantas tarefas rodam ao mesmo tempo

# A grade vem do config (bloco `binning:`), nao daqui: datasets diferentes tem bordas e
# quantidades de bins diferentes. `run.py grid` imprime "<n_et> <n_eta>".

# O job roda no nó de computação, que não herda confiavelmente o ambiente do nó de login: o
# interpretador do venv é resolvido por caminho absoluto e o diretório de trabalho do job é
# fixado no repositório (os caminhos de config, data/ e results/ são relativos a ele).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$REPO_DIR/neuralnet-env/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERRO: interpretador nao encontrado em '$PYTHON'." >&2
    echo "      Crie o ambiente com 'make venv' (dentro de uma alocacao, nao no no de login)" >&2
    echo "      ou indique outro com PYTHON=/caminho/para/python $0 ..." >&2
    exit 1
fi

read -r N_ET_BINS N_ETA_BINS < <(cd "$REPO_DIR" && "$PYTHON" ai/run.py grid --config "$CONFIG_FILE" --format shape 2>/dev/null | tail -1)
if [ -z "$N_ET_BINS" ] || [ -z "$N_ETA_BINS" ]; then
    echo "ERRO: falha ao ler a grade de '$CONFIG_FILE'." >&2
    echo "      Execute '$PYTHON ai/run.py grid --config $CONFIG_FILE' para ver o erro." >&2
    exit 1
fi

# n_splits e n_inits saem do mesmo config; sao eles que dizem quantos treinos existem por bin.
read -r N_FOLDS N_INITS < <(cd "$REPO_DIR" && "$PYTHON" -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from ai.run import load_config
c = load_config(sys.argv[1])
print(int(c.get('n_splits', 5)), int(c.get('n_inits', 1)))
" "$CONFIG_FILE")
if [ -z "$N_FOLDS" ] || [ -z "$N_INITS" ]; then
    echo "ERRO: falha ao ler n_splits/n_inits de '$CONFIG_FILE'." >&2
    exit 1
fi

N_BINS=$(( N_ET_BINS * N_ETA_BINS ))
N_TASKS=$(( N_BINS * N_FOLDS * N_INITS ))

# Índices do array: 0..N-1, opcionalmente com throttle (%N) para não ocupar todas as GPUs.
ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
if [ -n "$MAX_CONCURRENT" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

# Array vazio quando GRES nao esta definido: o --gres nao chega na linha de comando.
GRES_FLAG=()
if [ -n "$GRES" ]; then
    GRES_FLAG=(--gres="$GRES")
fi

echo "====================================================================="
echo "Grade Et x Eta submetida como job array: um treino por tarefa"
echo "Grade: ${N_ET_BINS}x${N_ETA_BINS} = ${N_BINS} regioes"
echo "Treinos: ${N_BINS} x ${N_FOLDS} folds x ${N_INITS} inits = ${N_TASKS} (array ${ARRAY_SPEC})"
echo "Arquivo de Configuração: $CONFIG_FILE"
echo "Partition: $PARTITION${GRES:+ (gres: $GRES)}"
echo "Interpretador: $PYTHON"
echo "====================================================================="

# Pasta para os logs do SLURM (arquivos .out e .err)
LOG_DIR="$REPO_DIR/results/logs_slurm"
mkdir -p "$LOG_DIR"

# Etapa 1: um treino por tarefa. O \$ é escapado deliberadamente: SLURM_ARRAY_TASK_ID só
# existe quando a tarefa roda no nó, portanto a aritmética que decompõe o índice em
# (et, eta, fold, init) é avaliada lá, e não na submissão. A decomposição é a inversa de
#   id = ((et * N_ETA + eta) * N_FOLDS + (fold-1)) * N_INITS + (init-1)
# de modo que tarefas do mesmo bin ficam contíguas e um throttle (%N) mantém juntas as que
# compartilham o mesmo carregamento de dados.
TRAIN_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 "${GRES_FLAG[@]}" \
     --array="$ARRAY_SPEC" \
     --job-name="train_one" \
     --output="${LOG_DIR}/train_%A_%a.out" \
     --error="${LOG_DIR}/train_%A_%a.err" \
     --wrap="i=\$SLURM_ARRAY_TASK_ID; \
             init=\$(( i % $N_INITS + 1 )); \
             i=\$(( i / $N_INITS )); \
             fold=\$(( i % $N_FOLDS + 1 )); \
             i=\$(( i / $N_FOLDS )); \
             eta=\$(( i % $N_ETA_BINS )); \
             et=\$(( i / $N_ETA_BINS )); \
             echo \"Task \$SLURM_ARRAY_TASK_ID -> et\${et}_eta\${eta} fold \${fold} init \${init}\"; \
             $PYTHON ai/run.py train --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta --fold \$fold --init \$init") || {
    echo "ERRO: sbatch recusou o array de treino; nada foi enfileirado." >&2
    exit 1
}

# Etapa 2: uma tarefa por regiao. So pode rodar quando TODOS os inits terminaram, porque
# `select` compara os n_inits checkpoints de cada fold para escolher o vencedor.
EVAL_JOB_ID=$(sbatch --parsable -p "$PARTITION" --chdir="$REPO_DIR" -N 1 "${GRES_FLAG[@]}" \
     --array="0-$(( N_BINS - 1 ))" \
     --dependency="afterok:${TRAIN_JOB_ID}" \
     --job-name="select_eval" \
     --output="${LOG_DIR}/eval_%A_%a.out" \
     --error="${LOG_DIR}/eval_%A_%a.err" \
     --wrap="et=\$(( SLURM_ARRAY_TASK_ID / $N_ETA_BINS )); \
             eta=\$(( SLURM_ARRAY_TASK_ID % $N_ETA_BINS )); \
             echo \"Task \$SLURM_ARRAY_TASK_ID -> et\${et}_eta\${eta}\"; \
             $PYTHON ai/run.py select --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta && \
             $PYTHON ai/run.py evaluate --config $CONFIG_FILE --et-bin \$et --eta-bin \$eta") || {
    echo "ERRO: sbatch recusou o array de avaliacao; cancele o treino com 'scancel ${TRAIN_JOB_ID}'." >&2
    exit 1
}

# Etapa 3: o tabelão agrega todas as regiões, então espera o array de avaliação inteiro.
REPORT_JOB_ID=$(sbatch --parsable --chdir="$REPO_DIR" -N 1 \
     --dependency="afterok:${EVAL_JOB_ID}" \
     --job-name="report" \
     --output="${LOG_DIR}/report_%j.out" \
     --error="${LOG_DIR}/report_%j.err" \
     --wrap="$PYTHON ai/run.py report --config $CONFIG_FILE") || {
    echo "ERRO: sbatch recusou o job do tabelão; cancele a grade com 'scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID}'" >&2
    echo "      ou execute o tabelão manualmente depois: $PYTHON ai/run.py report --config $CONFIG_FILE" >&2
    exit 1
}

echo "====================================================================="
echo "Treino:   array ${TRAIN_JOB_ID} (${N_TASKS} tarefas, um treino cada)."
echo "Select+avaliacao: array ${EVAL_JOB_ID} (${N_BINS} tarefas), apos o treino."
echo "Tabelao:  job ${REPORT_JOB_ID}, apos a avaliacao."
echo "Use 'squeue -u \$USER' para monitorar."
echo "Cancelar tudo: 'scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID} ${REPORT_JOB_ID}'."
echo "Uma tarefa isolada: 'scancel ${TRAIN_JOB_ID}_<indice>'."
echo "====================================================================="
echo "Resultados por região em 'results/<MODEL>/et<N>_eta<M>/', tabelão em"
echo "'results/<MODEL>/pd_table/' e logs em '${LOG_DIR}/'."
