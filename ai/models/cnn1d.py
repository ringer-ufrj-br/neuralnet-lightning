import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC 

# O LightningModule é uma classe base do Pytorch Lightning que encapsula toda a lógica do modelo, incluindo a definição da arquitetura, o cálculo da perda, a otimização e as métricas de avaliação.

class ModelCNN1D(pl.LightningModule): 
    def __init__(self, learning_rate=0.001):
        super().__init__() # Chama a classe pai (LightningModule) para inicializar corretamente o modelo.
        self.save_hyperparameters() # Salva os hiperparâmetros do modelo para referência futura.
        self.learning_rate = learning_rate # Define a taxa de aprendizado do modelo.

        # Arquitetura da Rede Neural Conolucional 1D:
        self.features = nn.Sequential(
            # Bloco Convolucional 1:
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)

            # Bloco Convolucional 2:
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Comprimento final do array será reduzido de 100 a 25 após 2 MaxPooling layers.
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), # Achata a saída do bloco de features para um vetor 1D porque a camada Linear espera uma entrada 1D.
            nn.Linear(in_features=64 * 25, out_features=128), # A entrada é 64 canais*25 comprimento final do array. A saída é 128 neurônios.
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        # Métricas de avaliação do modelo: 
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")

        self.criterion = nn.BCWithLogitsLoss() # Função de perda para classificação binária.

    # O forward() define como os dados passam pelo modelo durante a inferência. Ele recebe um tensor de entrada x e retorna a saída do modelo (predição).
    def forward(self, x):

        x = self.features(x) 
        x = self.classifier(x) 

        # Retorna o tensor resultante contendo as previsões brutas da rede.
        return x    

    def training_step(self, batch, batch_idx):
        # Desempacota o batch em dados de entrada (x) e rótulos (y).
        x, y = batch 

        # Adiciona uma dimensão extra ao tensor y para que ele tenha a forma correta para a função de perda, ou seja (batch_size, 1). 
        # A função unsqueeze(1) adiciona uma dimensão na posição 1, transformando o tensor de rótulo em um tensor 2D. 
        # Em seguida, converte o tensor para float.
        y = y.unsqueeze(1).float() 

        # Passa os dados de entrada pelo modelo para obter as predições (logits).
        logits = self(x) 

        # Calcula a perda entre as predições do modelo e os rótulos reais.
        loss = self.criterion(logits, y) 

        # Aplica a função sigmoide para obter probabilidades entre 0 e 1.
        preds = torch.sigmoid(logits) 

        self.train_accuracy(preds, y) # Atualiza a métrica de acurácia com as predições e os rótulos reais.
        self.train_auc(preds, y) # Atualiza a métrica de AUC com as predições e os rótulos reais.

        # O log registra as métricas de perda, acurácia e AUC para monitoramento durante o treinamento porque o Pytorch Lightning permite o registro de métricas para visualização em tempo real durante o treinamento.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auc', self.train_auc, on_step=True, on_epoch=True, prog_bar=True)

        return loss # Retorna a perda para que o Pytorch Lightning possa realizar a retropropagação e atualizar os pesos do modelo.

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze('1').float

        logits = self(x)
        loss = self.criterion(logits, y)
        preds = self.torch.sigmoid(logits)

        self.val_accuracy(preds,y)
        self.val_auc(preds, y)

        self.log('validation_loss', loss, prog_bar=True)
        self.log('validation_acc', self.val_accuracy, prog_bar=True)
        self.log('validation_auc', self.val_auc, prog_bar=True)

    # Configura o otimizador que atualizará os pesos da rede neural durante o treinamento. 
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
