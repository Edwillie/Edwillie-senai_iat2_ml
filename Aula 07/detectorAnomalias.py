import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

#Gerar um conjunto de dados aleatório
XNormais = 0.3 * np.random.randn(100, 2)
XOutliers = np.random.uniform(low = -4, high = 4, size = (20, 2))

X = np.r_[XNormais, XOutliers]

#Aplicar o modelo Isolation Forest
isolation = IsolationForest(contamination=0.15, random_state=50, n_estimators=1_000)
resultados = isolation.fit_predict(X)

print(XNormais)
print('---------------')
print(XOutliers)
print('---------------')
print(X)

print('---------------')
print(resultados)

cores = ['red' if p == -1 else 'blue' for p in resultados]

plt.scatter(X[:,0], X[:,1], c=cores)
plt.title("Anomalias detectadas")
plt.show()