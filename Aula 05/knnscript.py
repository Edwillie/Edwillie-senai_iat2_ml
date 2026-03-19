#Modulos e Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

#Usando dados de um dataset pequeno
data = {
    'tempAmbiente':[20,21,22,25,30,31,20,22,35,36],
    'rotacaoRPM':[1500,1520,1840,1600,1800,1850,1510,1490,2100,2200],
    'falha': [0,0,0,0,1,1,0,0,1,1]
}

df = pd.DataFrame(data)

#Dividir os dados
X = df[['tempAmbiente', 'rotacaoRPM']]
Y = df['falha']

#Normalizar os dados
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

#Dividir o dataset para treino e teste
X_Treino, X_Teste, Y_treino, Y_Teste = train_test_split(X_normalizado, Y, test_size= 0.2 )

knn_class = KNeighborsClassifier(n_neighbors=1)
knn_class.fit(X_Treino, Y_treino)


url = "http://bit.ly/knnClass"
df = pd.read_csv(url)