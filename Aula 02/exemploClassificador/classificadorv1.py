# 1. Importar as bibliotecas necessárias
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib as salvar

# 2. Carregar os dados (Simulando nossos dados industriais multivariados)
dados = load_breast_cancer()
X = dados.data  # As "Features" (características das células/sensores)
y = dados.target # O "Target" (0: Maligno, 1: Benigno)

# 3. Pré-processamento: Normalização (ESSENCIAL para Redes Neurais/MLP)
# Redes neurais não funcionam bem com números em escalas muito diferentes.
scaler = StandardScaler()
X_escalonado = scaler.fit_transform(X)

# 4. Dividir em Treino (80%) e Teste (20%)
# O modelo estuda o treino e faz a prova no teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_escalonado, y, test_size=0.20, random_state=42
)

# 5. Criar e Treinar o Perceptron Multicamada (MLP)
# Vamos criar uma rede com 2 camadas escondidas (10 neurônios cada)
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10), 
    max_iter=500, 
    random_state=42,
    verbose=False # Mude para True se quiser ver o erro caindo por época
)

mlp.fit(X_treino, y_treino) # Aqui o robô está aprendendo!

# 6. Realizar as Predições
predicoes = mlp.predict(X_teste)

# 7. Avaliar o comportamento (Métricas de Precisão)
print("--- Matriz de Confusão ---")
print(confusion_matrix(y_teste, predicoes))
print("\n--- Relatório de Classificação ---")
print(classification_report(y_teste, predicoes))

# 8. Salvar o modelo treinado
salvar.dump(mlp, 'modeloMLP.pkl') 
salvar.dump(scaler, 'scaler.pkl')