# 1. Importar as bibliotecas necessárias
import joblib as salvar
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 2. Carregar o modelo
print("Carregando o modelo.")
vendaDeImoveis = fetch_california_housing()

X = vendaDeImoveis.data[:, :6]
Y = vendaDeImoveis.target

# 3. Dividir a bas em treino/teste
XTreino, XTeste, YTreino, YTeste = train_test_split(X, Y, test_size=0.2, random_state=50)

# 4. Normalizaçao da escala
scaler = StandardScaler()
XTreinoNormalizado = scaler.fit_transform(XTreino)
XTeste = scaler.transform(XTeste)

# 5. Iniciando a Regressao
mlp = MLPRegressor(
    hidden_layer_sizes=(30,15),
    max_iter=500,
    random_state=50,
    verbose=False
) 