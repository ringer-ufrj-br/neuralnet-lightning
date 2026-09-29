# Rodando no cluster

Como submeter a grade com `slurm_bins.sh` sem desperdiçar o cluster. O resumo: **as MLPs
treinam em CPU**, e a submissão em CPU precisa de três variáveis de ambiente que o script não
define sozinho.

```bash
PARTITION=cpu SBATCH_MEM_PER_NODE=2G \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
./scripts/slurm_bins.sh ai/configs/mlp_mc21.yaml
```

E no config:

```yaml
accelerator: cpu
```

---

## Por que CPU

A MLP de anéis é rasa (50 entradas e poucas camadas pequenas): não há trabalho suficiente por
camada para aproveitar uma GPU nem vários núcleos. Ela escala mal em threads e muito bem em
tarefas: a grade são 625 treinos independentes (25 regiões × 5 folds × 5 inicializações), então
o melhor é rodar o máximo de tarefas ao mesmo tempo, cada uma com 1 núcleo.

A partition `cpu` tem 120 núcleos em 12 nós (`caloba10`–`caloba21`, 10 núcleos e 28 GB cada).
Com as configurações abaixo, a grade mc21 inteira (treino, avaliação e tabelão) roda em ~30 min,
com 120 tarefas simultâneas.

GPU só vale para as redes maiores (CNN2D, Fused): nesse caso use a partition `gpu`, que é o
default do script.

## As três configurações

**1. Memória: `SBATCH_MEM_PER_NODE=2G`.** É a mais importante, e falha em silêncio. O cluster
tem `DefMemPerNode=UNLIMITED` com `CR_CORE_MEMORY`: uma tarefa que não pede memória reserva os
28 GB do nó inteiro, e nenhuma outra entra ali. Resultado: 1 tarefa por nó, 12 rodando numa
partition de 120 núcleos, e a grade leva horas em vez de minutos. Nada dá erro; o array só
anda devagar. O pico por tarefa é ~1,0 GB na maior região do mc21 (et3_eta0, 354 mil eventos;
~0,7 GB disso é só importar torch e Lightning), então 2G dá folga e cabem 10 tarefas por nó.
`SBATCH_MEM_PER_NODE` é o equivalente em variável de ambiente do `--mem`, que o script não
recebe como argumento.

**2. Threads: `OMP_NUM_THREADS=1`** (e `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, que nem
sempre seguem o OMP). O SLURM entrega 1 núcleo por tarefa, mas o torch lê a contagem de núcleos
da máquina (10) e abre 10 threads dentro desse núcleo. A troca de contexto custa mais que o
trabalho útil: uma tarefa caiu de ~6 min para ~1 min 20 s quando as threads foram limitadas a 1.

**3. Partition: `PARTITION=cpu`.** Sem isso o script usa a `gpu`, que é o default dele.

O `accelerator: cpu` no config deixa explícito o que roda: o `--wrap` do script não passa
`--accelerator`, então o config é o único lugar para isso.

## Conferindo uma grade

Enquanto roda, deve aparecer ~10 tarefas por nó. Se aparecer 1, a memória não foi aplicada:

```bash
squeue -h -t RUNNING -u $USER -o "%N" | sort | uniq -c
```

Depois que termina, memória e tempo de cada tarefa:

```bash
sacct -j <jobid> --units=M -o JobID%14,MaxRSS,Elapsed,State%12
```

O `sacct` pode subestimar o pico: o SLURM amostra a memória a cada 30 s
(`JobAcctGatherFrequency = 30`), e o pico real dura poucos segundos (a leitura e a montagem da
matriz de features, no começo da tarefa). O limite de memória, porém, vale para esse instante.
Quando o código ainda convertia o dataframe para pandas, o `sacct` mostrava 1,09 GB para um pico
real de ~1,25 GB; conte com ~15% a mais do que ele mostra. Se o valor passar de ~1,6G (dataset
maior, mais colunas), aumente `SBATCH_MEM_PER_NODE`: abaixo do pico a tarefa morre por falta de
memória, e muito acima volta-se a desperdiçar o nó.

## Pegadinhas

- **No máximo 1001 tarefas por array** (`MaxArraySize = 1001`). 25 × 5 × 5 = 625 cabe; com 10
  folds seriam 1250, e o `sbatch` recusa com `Invalid job array specification` sem enfileirar
  nada.
- **Comece de uma pasta de resultados limpa** ao mudar `n_splits` ou a versão do código. O
  `evaluate` escolhe entre todos os `checkpoints/fold_*.json` que encontrar, então arquivos de
  uma rodada anterior entram na escolha sem aviso.
- **`sstat` não mostra nada** neste cluster: a memória de uma tarefa em andamento não é
  visível. Use o `sacct` depois que ela termina, lembrando que ele subestima o pico.
- **O aviso do Lightning sobre `srun`** é ruído: com um processo por tarefa, não há nada
  configurado errado.
