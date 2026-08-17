import pandas as pd
import numpy as np

class PreProcessCNN1D:
    def __init__(self):
        # Inicializa o pré-processador para a CNN1D
        # feature_columns lista com o nome das 100 colunas de entrada da rede

        self.num_columns = 100
        self.feature_columns = [f"cl_ring_{i}" for i in range(self.num_columns)]
        
    def transform(self, df): 
    # Transforma as colunas de interesse em Arrays do Numpy
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Faltam {len(missing)} colunas no DataFrame")
        print(f"Extraindo {self.num_columns} colunas do DataFrame ...")

        # Converte para Numpy (Batch, 100) apenas as colunas de interesses recortadas do dataframe
        X = df[self.feature_columns].values.astype(np.float32)

        X = np.where(X == -999, 0, X)

        # A função clip limita os valores de X, substituindo valores negativos por 0.
        # A função np.log1p aplica uma função log a cada elemento de X
        # Reduzem o impacto de outliers
        X = np.log1p(np.clip(X, 0, None))

        # Transforma o shape de (Batch, 100) 2D, para (Batch, 1, 100) 3D, que é o formato para o Pytoch 
        X = np.expand_dims(X, axis=1)

        return X 

    def get_labels(self, df, label_column = 'label'):
        # Retorna os labels
        if label_column in df.columns:
            return df[label_column].values.astype(np.float32)
        return None