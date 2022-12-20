import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier
from sklearn.preprocessing import LabelEncoder

# Base de dados
np.random.seed(123)
torch.manual_seed(123)

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

encoder = LabelEncoder()
classe = encoder.fit_transform(classe)

np.unique(classe)

previsores = previsores.astype('float32')
classe = classe.astype('int64')

# Construção do modelo
class classificador_torch(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense0 = nn.Linear(4,16)
        self.activation0 = nn.ReLU()
        self.dense1 = nn.Linear(16,16)
        self.activation1 = nn.ReLU()
        self.dense2 = nn.Linear(16,3)

    def forward(self, X):
        X = self.dense0(X)
        X = self.activation0(X)
        X = self.dense1(X)
        X = self.activation1(X)
        X = self.dense2(X)
        return X

classificador_sklearn = NeuralNetClassifier(
    module = classificador_torch,
    criterion = torch.nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    max_epochs = 1000,
    batch_size = 10,
    train_split = False
)

# Validação
resultados = cross_val_score(classificador_sklearn, previsores, classe, cv = 5, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()

print(f'Média: {media} ----- Desvio: {desvio}')


print('Resultado: ', resultados)