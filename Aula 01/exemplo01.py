from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Carregando o modelo padronizado e validado
dados = load_breast_cancer()
X = dados.data   # Os atributos - features
Y = dados.target # Os rotulos/alvos - target

#Pre-processamento dos dados do dataset
escala = StandardScaler()
X_escalonado = escala.fit_transform(X)

#Selecionar os dados de treinamento/teste (80%/20%)
### argumento do metodo train_test_split
### 1- dados processado / normalizado
### 2- conjunto alvo
### test_size => porcentagem de dados de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_escalonado, Y, test_size= 0.2)

#Treinamento da MLP (Rede Neural)
mlp = MLPClassifier(hidden_layer_sizes=(10,10),
                    max_iter=500,
                    random_state=42,
                    verbose=True)

mlp.fit(X_treino, Y_treino)

predicao = mlp.predict(X_teste)

#Apresentando resultados
print("Matriz de confusao =>")
print(confusion_matrix(Y_teste, predicao))


print("Relatório Classificacao =>")
print(classification_report(Y_teste, predicao))



