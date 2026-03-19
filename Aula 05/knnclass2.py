#Modulos e Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

url = "http://bit.ly/knnClass"
df = pd.read_csv(url)

X = df[['Air Temperatura [K]', 'Process Temperature [K]', 'Rotational Speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
Y = df['MAchine failure']

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)
X_Treino, X_Teste, Y_treino, Y_Teste = train_test_split(X_normalizado, Y, test_size= 0.3 )

#Aplicando KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_Treino, Y_treino)

y_Pred = knn.predict(X_Teste)

print("=Resultados=")
