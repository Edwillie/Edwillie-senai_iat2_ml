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




