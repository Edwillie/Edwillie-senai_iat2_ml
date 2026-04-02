import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import pandas as pd

nomeArquivo = "germany_energy.csv"
if not os.path.exists(nomeArquivo):
    print("Arquivo Inexistente")
    exit()
else:
    print("Inicializando dataset...")
    df = pd.read_csv(nomeArquivo)

#Pre-tratamento dos dados
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Consumption'])

df['DayOfWeek'] = df['Date'].dt.day_of_week
df['Month'] = df['Date'].dt.month

features = ['Consumption', 'DayOfWeek', 'Month']

X = df[features]
scaler = StandardScaler()
XNormalizado = scaler.fit_transform(X)

print(XNormalizado)

#Aplicar o modelo Isolation Forest
modelo = IsolationForest(contamination=0.15, random_state=50, n_estimators=500)
df['isAnomaly'] = modelo.fit_predict(XNormalizado)

anomaly = df[df['isAnomaly'] == -1]
print(anomaly)
plt.figure()
plt.plot(df['Date'], df['Consumption'], color='gray', label='Dados de Consumo')
plt.scatter(anomaly['Date'], anomaly['Consumption'], color='red', edgecolors='black', label='Anomalias')
plt.legend()
plt.title("Dados Consumo de Energia")
plt.xlabel("Ano")
plt.ylabel("Consumo (GWh)")
plt.grid(True, alpha=0.3)
plt.show()