# neuralnet-lightning

Orquestrador e pipeline de treinamento de redes neurais (baseado em PyTorch / PyTorch Lightning) voltado para análise de dados do ATLAS (CERN).

---

## 📌 Visão Geral

O projeto automatiza o fluxo de carregamento de dados (ex: arquivos Parquet), pré-processamento, criação e treinamento de modelos de deep learning (como `CNN2D`), validação por K-Fold cross-validation e avaliação de desempenho.

---

## 📁 Estrutura Principal do Projeto

- **`ai/`**: Contém os módulos de inteligência artificial (pipelines, modelos, carregadores de dados e pré-processadores).
  - `ai/run.py`: Script principal de entrada (entrypoint) para orquestrar os experimentos.
  - `ai/pipeline/`: Pipelines de treinamento e avaliação.
  - `ai/models/`: Definições das arquiteturas das redes neurais.
- **`config.yaml`**: Arquivo YAML com as configurações e hiperparâmetros do experimento.
- **`data/`**: Diretório destinado aos conjuntos de dados (arquivos Parquet/ROOT).
- **`results/`**: Diretório onde são salvos os relatórios, métricas e checkpoints dos modelos.
- **`Makefile` & `activate.sh`**: Scripts utilitários para criação e ativação do ambiente virtual Python.

---

## ⚙️ Pré-requisitos e Instalação

1. **Criar o ambiente virtual e instalar as dependências:**
   ```bash
   make venv
   ```
2. **Ativar o ambiente virtual:**
   ```bash
   source activate.sh
   # ou: source neuralnet-env/bin/activate
   ```

---

## ⚙️ Configuração (`config.yaml`)

As configurações da execução são definidas no arquivo `config.yaml`. Exemplo de parâmetros suportados:

```yaml
model: "CNN2D"                                     # Modelo a ser utilizado
data_path: data/parquet/mc25_13TeV...parquet       # Caminho para os dados
max_files: 100                                     # Quantidade máxima de arquivos
label_col: "label"                                 # Coluna de rótulo (label)
max_epochs: 200                                    # Número máximo de épocas
batch_size: 64                                     # Tamanho do batch
learning_rate: 0.001                               # Taxa de aprendizado
patience: 20                                       # Paciência para Early Stopping
use_kfold: true                                    # Ativar validação cruzada K-Fold
n_splits: 3                                        # Número de splits do K-Fold
test_size: 0.15                                    # Proporção do conjunto de teste
```

---

## 🚀 Como Executar (`run.py`)

A execução do treinamento é realizada através do script [ai/run.py](file:///home/joao.gomes/LPS/cern/ai/run.py).

### 1. Execução Padrão
Executa o fluxo completo usando as configurações padrão do arquivo `config.yaml`:
```bash
python ai/run.py
```

### 2. Especificando um arquivo de configuração personalizado
Para utilizar outro arquivo YAML de configuração:
```bash
python ai/run.py --config caminho/para/seu_config.yaml
```

### 3. Executando um Fold específico (Paralelização / SLURM)
Para rodar apenas um fold específico (por exemplo, no Fold 0):
```bash
python ai/run.py --fold 0
```
*Útil para jobs em paralelo em ambientes de HPC utilizando SLURM.*

---

## ⚖️ Estratégia de Balanceamento de Classes (Weighted Loss)

O repositório utiliza **função de custo ponderada** (`nn.BCEWithLogitsLoss(pos_weight=pos_weight)`) em substituição ao *undersampling* aleatório, preservando 100% dos dados originais e evitando o descarte de informações relevantes.

### 1. Cálculo Dinâmico de `pos_weight`
O peso da classe positiva é calculado estritamente a partir das amostras do conjunto de **treino** de cada execução/fold (evitando qualquer vazamento de dados de teste ou validação):
$$\text{pos\_weight} = \frac{N_{\text{negativos}}}{N_{\text{positivos}}}$$

- **No K-Fold**: A cada fold, o `pos_weight` é recalculado para os dados específicos daquele split (`train_ids`).
- **Buffer PyTorch**: O `pos_weight` é registrado como buffer (`register_buffer("pos_weight", ...)`) no `LightningModule`, garantindo que acompanhe o dispositivo (CPU/GPU) sem ser otimizado como parâmetro.
- **Consistência**: A partição estratificada com `random_state=42` garante que múltiplos modelos (`CNN2D`, `MLP`) comparem seus resultados sob os mesmos splits e pesos.

### 2. Métricas de Avaliação para Dados Desbalanceados
Com a preservação do desbalanceamento natural, a acurácia global deixa de ser a única métrica de referência. O pipeline monitora e registra:
- **Precision, Recall e F1-Score** (por classe)
- **AUC-ROC** (Área sob a curva ROC)
- **AUC-PR / Average Precision** (Área sob a curva Precision-Recall — métrica primária em forte desbalanceamento)
- **Curva Precision-Recall (`pr_curve.pdf`)** e **Matriz de Confusão (`confusion_matrix.pdf`)**

### 3. Calibração do Limiar de Decisão (*Threshold*)
Como o modelo é treinado com penalidade assimétrica (`pos_weight`), o limiar padrão de `0.5` pode favorecer o recall em detrimento da precisão. Recomenda-se analisar a curva PR gerada em `results/<MODEL>/plots/pr_curve.pdf` para selecionar o ponto de operação ideal (ajustável via parâmetro `threshold` no `config.yaml`).

---

## 🧹 Limpeza do Ambiente

Para remover o ambiente virtual e arquivos compilados `.pyc`:
```bash
make clean
```