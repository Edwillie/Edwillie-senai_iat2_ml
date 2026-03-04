import itertools

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

COMPONENTES = {
    'Painel': {0: 0.05, 1:0.99},
    'Filtro': {0: 0.00, 1:1.00},
    'Tanque': {0: 0.00, 1:1.00}
}

def calcularProbVazao(pPainel, pFiltro, pTanque):
    p = (COMPONENTES['Painel'] [pPainel] *
         COMPONENTES['Filtro'] [pFiltro] *
         COMPONENTES['Tanque'] [pTanque] 
        )
    return round(p, 4)


#Função que define os parametros para a tecnica bayesiana
def gerarResposta():
    tabelaCombinacoes = list(itertools.product([0, 1], repeat = 3))
    probSim = []
    probNao = []

    for iPainel, iFiltro, iTanque in tabelaCombinacoes:
        probTerVazao = calcularProbVazao(pPainel=iPainel, pFiltro=iFiltro, pTanque=iTanque)
        probSim.append(probTerVazao)
        probNao.append(1 - probTerVazao)

    return TabularCPD(variable = 'Vazao', variable_card = 2, values = [probNao, probSim], evidence=['Painel', 'Filtro', 'Tanque'], evidence_card = {2, 2, 2} )

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

    #Chamando a função que aplica a técnica de redes Bayesianas
    cpdVazao = gerarResposta()

    cpdSensor = TabularCPD('Sensor', 2, [0.95, 0.02], evidence='Tanque', evidence_card=[2])

    modelo.add_cpds(cpdPainel, cpdFiltro, cpdTanque, cpdVazao, cpdSensor)

    assert modelo.check_model()
    return modelo


if __name__ == '__main__':
    criarSistema()