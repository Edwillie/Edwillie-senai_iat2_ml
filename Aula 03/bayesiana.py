from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def criarSistema():
    # Configuração do modelo
    modelo = DiscreteBayesianNetwork([('Painel','Vazao'), ('Filtro', 'Vazao'), ('Tanque', 'Vazao'), ('Motor', 'Vazao'), ('Fluido', 'Vazao'), ('Tanque', 'Sensor')])

    # Criar estrutura de interação direta com o problema
    cpdPainel = TabularCPD(variable='Painel', values=[0, 1])
    cpdFiltro = TabularCPD(variable='Filtro', values=[0, 1])
    cpdTanque = TabularCPD(variable='Tanque', values=[0, 1])

def sistema():
    pass