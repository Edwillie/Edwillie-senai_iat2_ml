from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def criarSistema():
    # Configuração do modelo
    modelo = DiscreteBayesianNetwork(
        [
            ('Painel','Vazao'),
            ('Filtro','Vazao'),
            ('Tanque','Vazao'),
            ('Tanque','Sensor')
        ]
    )
    # Criar estruturas de interação 
    # direta com o problema
    cpdPainel = TabularCPD(variable='Painel', variable_card=2, values= [[0.01],[0.99]] )
    cpdFiltro = TabularCPD(variable='Filtro',variable_card=2, values=[[0.1],[0.9]])
    cpdTanque = TabularCPD(variable='Tanque',variable_card=2, values=[[0.05],[0.95]])