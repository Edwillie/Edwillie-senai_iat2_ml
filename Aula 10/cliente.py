import streamlit as st
import seaborn as sb
import joblib
import numpy as np
import matplotlib.pyplot


st.set_page_config(page_title="Análise por KDE", layout="wide")
@st.cache_resource

def loadModel():
    return joblib.load('modelo_total.pkl')

try:
    data = loadModel()
    kde = data['kde']
    scaler = data['scaler']
    thres = data['thres']
    caracteristicas = data['caracteristicas']
    trainScores = data.get('trainScores', [])
    st.title("Validação do Compressor")
    st.info("Configuração do modelo - KDE")
except:
    st.error(f"Erro ao carregar o modelo")
finally:
    st.stop()