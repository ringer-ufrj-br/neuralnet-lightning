#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) & Data Pipeline Test
# Carregamento de dados, geração de labels, pré-processamento para a CNN 2D e salvamento do CSV (features + label).

# In[ ]:


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionando o diretório raiz ao path
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.loader.loader import DataLoader
from ai.label.label_generator import label_from_path
from ai.preprocess.cnn2d import PreprocessCNN2D


# In[ ]:


# Diretório para salvar resultados da EDA
results_dir = os.path.join(PROJECT_ROOT, "results", "eda")
os.makedirs(results_dir, exist_ok=True)

# 1. Carregamento dos dados Parquet
data_path = os.path.join(PROJECT_ROOT, "data", "parquet", "**", "*.parquet")
loader = DataLoader(data_path=data_path, max_files=1) # Limitado a 1 arquivo para execução rápida
df = loader.execute()

# 2. Geração de Labels
label_col = 'label'
df[label_col] = df["file_path"].apply(label_from_path)
if 'file_path' in df.columns:
    df.drop(columns=['file_path'], inplace=True)


# In[ ]:


# 3. Pré-processamento das matrizes do calorímetro (Estruturação 4D para CNN 2D)
preprocessor = PreprocessCNN2D()
X = preprocessor.transform(df)
Y = preprocessor.get_labels(df, label_col=label_col)

print(f"\n--- Resumo dos dados processados ---")
print(f"Shape de X (Batch, Canais, Altura, Largura): {X.shape}")
print(f"Shape de Y (Labels): {Y.shape}")
print(f"Distribuição das classes (0=JF17, 1=Zee): {np.bincount(Y.astype(int))}")

# 4. Salvar CSV com entradas do modelo nas primeiras colunas e o label na última coluna
X_flat = X.reshape(X.shape[0], -1) # Achata (N, 7, 7, 15) -> (N, 735)
feature_cols = [f"feat_{i}" for i in range(X_flat.shape[1])]
df_csv = pd.DataFrame(X_flat, columns=feature_cols)
df_csv[label_col] = Y

csv_path = os.path.join(results_dir, 'model_input.csv')
df_csv.to_csv(csv_path, index=False)
print(f"\nCSV exportado com sucesso: {csv_path}")
print(f"Formato do CSV gerado: {df_csv.shape} (735 features + 1 label)")
df_csv.head()

