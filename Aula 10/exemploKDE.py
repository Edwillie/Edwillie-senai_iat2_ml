# bibliotecas necessarias
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def scoreWithProgress(modelo, dados, batchSize = 500):
    scores = []
    for i in tqdm(range(0,len(dados),batchSize)
                  ,desc="Calculando Densidade"):
        pacote = dados[i:i + batchSize]
        scores.extend(modelo.score_samples(pacote))
    return np.array(scores)

#Carregando os dados do dataset
print("=== Iniciando treinamento ===")
caracteristicas = ['TP3','H1','Motor_current']
df = pd.read_csv('metro.csv',usecols=caracteristicas)

#Dividindo os dados
trainSize = 50000

XtreinoBruto = df.iloc[:int(len(df) * 0.8)].sample(n=trainSize, random_state=50)
XtreinoBrutoFinal = XtreinoBruto[caracteristicas]
Xteste = df.iloc[:trainSize][caracteristicas]
print(XtreinoBrutoFinal)
#Normalizacao
scaler = StandardScaler()
#Trocar para RobustScaler
XtreinoNormalizado = scaler.fit_transform(XtreinoBrutoFinal)
XtesteNormalizado = scaler.fit_transform(Xteste)

#Implementando o modelo
modelo = KernelDensity(kernel='gaussian', algorithm='ball_tree', leaf_size=40, bandwidth=0.5)
modelo.fit(XtreinoNormalizado)

trainScores = scoreWithProgress(modelo, XtreinoNormalizado)

threshold = np.percentile(trainScores, 2)
testScores = scoreWithProgress(modelo, XtesteNormalizado)

Y = [1 if s < threshold else 0 for s in testScores]

normais = [x for x in Y if x == 1]
Anormais = [x for x in Y if x == 0]

plt.figure()
plt.scatter(np.arange(len(normais)), normais)
plt.scatter(np.arange(len(Anormais)), Anormais)
plt.show()

# Parar por aqui...

