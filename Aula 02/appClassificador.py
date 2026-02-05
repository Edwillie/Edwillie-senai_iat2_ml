import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Configurando a Janela
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

# 1. Carregar Modelo e Scaler
try:
    modelo = joblib.load('modeloMLP.pkl')
    scaler = joblib.load('scaler.pkl')

    print("Sistema Carregado. ")
except:
    print("Erro ao carregar os modelos.")
    exit()

# 2. Importando dados do paciente
print("Selecionando o arquivo do paciente.")
caminhoCSV = filedialog.askopenfilename(title="Selecione o arquivo do paciente", filetypes=[("Arquivos CSV", "*.csv")])

if caminhoCSV:
    dadosPaciente = pd.read_csv(caminhoCSV)

    if dadosPaciente.shape[1] != 30:
        print("Erro do leiaute de paciente")
        exit()

    dadosEscalonados = scaler.transform(dadosPaciente.values)

    # 3. Predição com base nos dados do paciente
    predicao = modelo.predict(dadosEscalonados)
    probabilidade = modelo.predict_proba(dadosEscalonados).max()

    if predicao[0] == 1:
        print("Dados de Paciente Benigno")
    else:
        print("Dados do Paciente Maligno")