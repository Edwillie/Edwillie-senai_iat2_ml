import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Carregar o modelo
print("Carregando o modelo.")
vendaDeImoveis = fetch_california_housing()

X = vendaDeImoveis.data[:,:6]
Y = vendaDeImoveis.target

# 2. Dividir a base em treino/teste

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size= 0.2, random_state= 50
) 

# 3. Normalização
scaler = StandardScaler()
X_train_normalizado = scaler.fit_transform(X_train)
X_test_normalizado = scaler.transform(X_test)

# 4. Iniciando a regressão
mlp = MLPRegressor(
    hidden_layer_sizes= (30,15),
    max_iter= 1000,
    random_state= 42
)

# Realizando o treinamento do MLP
mlp.fit(X_train_normalizado, Y_train)

# Realizando a predicao
pred = mlp.predict(X_test_normalizado)

# Salvando os modelos
joblib.dump(mlp, 'modelo_regressao.pkl')
joblib.dump(scaler, 'modelo_normalizado.pkl')

# Visualizando os resultados
quantidade = 60
Y_real = Y_test[:quantidade]
Y_previsto = pred[:quantidade]

ids = range(len(Y_real))

# Criando espaco de trabalho da figura
plt.figure(figsize=(12,6))

# plotando o grafico
plt.plot(ids, Y_real, color='red', label="Real", marker="o", linestyle="-", linewidth=2)
plt.plot(ids, Y_previsto, color='blue', label="Previsto", marker="x", linestyle="--", linewidth=2)

plt.xlabel("ID da Casa")
plt.ylabel("Preço (USS$)")
plt.title("Valores de Imoveis")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()